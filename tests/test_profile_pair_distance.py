"""Run with: python3 -m unittest discover -s tests -p test_profile_pair_distance.py

Set MIN_PAIR_TEST_DEVICE=cuda:3 to select a GPU; CUDA tests skip without CUDA.
"""

import math
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import torch
except ModuleNotFoundError:
    torch = None
if torch is not None:
    from profile_pair_distance import profile_pair_distance, load_fvecs
from vecs_io import fvecs_write


@unittest.skipUnless(torch is not None, "Optional PyTorch dependency required")
class TestInput(unittest.TestCase):
    def test_valid_and_invalid_fvecs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base.fvecs"
            points = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
            fvecs_write(path, points)
            np.testing.assert_array_equal(load_fvecs(path), points)
            valid = path.read_bytes()
            invalid = [b"", valid[:-1], valid[:-4], valid[:12]]
            headers = np.frombuffer(valid, dtype="<i4").copy()
            headers[3] = 1
            invalid.append(headers.tobytes())
            headers[0] = 0
            invalid.append(headers.tobytes())
            for data in invalid:
                with self.subTest(data_size=len(data)):
                    path.write_bytes(data)
                    with self.assertRaises(ValueError):
                        load_fvecs(path)
            for value in [math.nan, math.inf, -math.inf]:
                points[0, 0] = value
                fvecs_write(path, points)
                with self.assertRaises(ValueError):
                    load_fvecs(path)


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA PyTorch required")
class TestPairProfileCUDA(unittest.TestCase):
    def test_against_cpu_exhaustive(self):
        rng = np.random.default_rng(42)
        cases = {
            "random": rng.normal(size=(37, 9)).astype(np.float32),
            "cross_block": np.array([[0, 0], [8, 7], [5, 3], [0.01, 0]], dtype=np.float32),
            "duplicate": np.array([[1, 2], [4, 5], [1, 2]], dtype=np.float32),
            "zeros": np.zeros((5, 3), dtype=np.float32),
            "two": np.array([[0, 0], [3, 4]], dtype=np.float32),
            "ties": np.arange(21, dtype=np.float32).reshape(-1, 1),
            "cancellation": np.array([[1e8, 0.01], [1e8, 0.002],
                                      [1e8, 0.003], [1e8, 0.03]], dtype=np.float32),
            "subnormal": np.array([[0], [np.nextafter(np.float32(0), np.float32(1))],
                                   [1]], dtype=np.float32),
        }
        for name, points in cases.items():
            distances = [math.fsum((float(a) - float(b)) ** 2
                                  for a, b in zip(points[i], points[j]))
                         for i in range(len(points)) for j in range(i + 1, len(points))]
            for block_size in (1, 2, 7, 64):
                for mode in ("float64", "float32", "tf32"):
                    with self.subTest(case=name, block_size=block_size, matmul=mode):
                        result = profile_pair_distance(
                            points, os.environ.get("MIN_PAIR_TEST_DEVICE", "cuda:0"),
                            block_size, 0, mode)
                        for extremum, expected in (("min", min(distances)), ("max", max(distances))):
                            self.assertTrue(math.isclose(result[extremum + "_squared_distance"], expected,
                                                         rel_tol=1e-14, abs_tol=0.0))
                            i, j = result[extremum + "_pair_ids"]
                            self.assertLess(i, j)
                            actual = math.fsum((float(a) - float(b)) ** 2
                                               for a, b in zip(points[i], points[j]))
                            self.assertEqual(actual, result[extremum + "_squared_distance"])
                        self.assertEqual(result["evaluated_unique_pairs"],
                                         len(points) * (len(points) - 1) // 2)
                        if min(distances):
                            self.assertEqual(result["delta"], math.sqrt(max(distances)) / math.sqrt(min(distances)))
                            self.assertEqual(result["delta_status"], "finite")
                        else:
                            self.assertIsNone(result["delta"])
                            self.assertEqual(result["delta_status"], "infinite" if max(distances) else "undefined")

    def test_compiled_reductions(self):
        rng = np.random.default_rng(123)
        device = os.environ.get("MIN_PAIR_TEST_DEVICE", "cuda:0")
        for points in (rng.normal(size=(35, 33)).astype(np.float32),
                       rng.integers(-8, 9, size=(35, 33)).astype(np.float32)):
            reference = profile_pair_distance(points, device, 13, 0)
            result = profile_pair_distance(points, device, 13, 0, "tf32", True)
            for key in ("min_squared_distance", "max_squared_distance", "delta", "evaluated_unique_pairs"):
                self.assertEqual(result[key], reference[key])

    def test_dense_compiled_candidates(self):
        # > 1M tied contenders: exercise FP64 fallback and bounded enumeration.
        points = np.full((1537, 1), 1.25, dtype=np.float32)
        result = profile_pair_distance(points, os.environ.get("MIN_PAIR_TEST_DEVICE", "cuda:0"),
                                       2048, 0, "tf32", True)
        self.assertEqual(result["min_distance"], 0)
        self.assertEqual(result["max_distance"], 0)
        self.assertEqual(result["delta_status"], "undefined")
        self.assertEqual(result["fp64_rescanned_blocks"], 1)
        self.assertEqual(result["evaluated_unique_pairs"], 1537 * 1536 // 2)

    def test_invalid_arguments(self):
        points = np.zeros((2, 3), dtype=np.float32)
        for values, options in [(points[:1], {}), (points.astype(np.float64), {}),
                                (points, {"block_size": 0}), (points, {"device": "cpu"}),
                                (points, {"progress_interval": -1})]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                profile_pair_distance(values, **options)


if __name__ == "__main__":
    unittest.main()
