from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate_queries import generate_queries
from vecs_io import bvecs_write, fvecs_read, fvecs_write


class GenerateQueriesTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.base = self.root / "base.fvecs"
        self.query = self.root / "query.fvecs"

    def test_sample_all_rows_and_add_iid_noise_across_output_chunks(self):
        rows = np.linspace(-1, 1, 5000, dtype="float32")[:, None].repeat(4, axis=1)
        fvecs_write(self.base, rows)
        original = self.base.read_bytes()
        generate_queries(self.base, self.query, len(rows), 0)
        sources = fvecs_read(self.query)
        np.testing.assert_array_equal(np.sort(sources[:, 0]), rows[:, 0])
        generate_queries(self.base, self.query, len(rows), 0.04)
        noise = fvecs_read(self.query) - sources
        self.assertAlmostEqual(float(noise.mean()), 0.0, delta=0.005)
        np.testing.assert_allclose(np.cov(noise.T), np.eye(4) * 0.04, atol=0.003)
        saved = self.query.read_bytes()
        generate_queries(self.base, self.query, len(rows), 0.04)
        self.assertEqual(self.query.read_bytes(), saved)
        self.assertEqual(self.base.read_bytes(), original)

    def test_invalid_settings_and_output_collision_preserve_files(self):
        fvecs_write(self.base, np.zeros((3, 2), dtype="float32"))
        self.query.write_bytes(b"previous output")
        original = self.base.read_bytes()
        for count, variance in [(0, 1), (4, 1), (1, -1), (1, float("nan")), (1, float("inf"))]:
            with self.subTest(count=count, variance=variance), self.assertRaises(ValueError):
                generate_queries(self.base, self.query, count, variance)
        with self.assertRaises(ValueError):
            generate_queries(self.base, self.base, 1, 0)
        self.assertEqual(self.base.read_bytes(), original)
        self.assertEqual(self.query.read_bytes(), b"previous output")
        # A failure while writing a later batch must also leave the old output intact.
        rows = np.zeros((5000, 2), dtype="float32")
        last_id = np.random.default_rng(42).choice(len(rows), len(rows), replace=False)[-1]
        rows[last_id, 0] = np.nan
        fvecs_write(self.base, rows)
        with self.assertRaises(ValueError):
            generate_queries(self.base, self.query, len(rows), 1)
        self.assertEqual(self.query.read_bytes(), b"previous output")
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_byte_base_produces_fractional_unclipped_float_queries(self):
        base = self.root / "base.bvecs"
        bvecs_write(base, np.zeros((50, 3), dtype="uint8"))
        generate_queries(base, self.query, 50, 0.25)
        queries = fvecs_read(self.query)
        self.assertEqual(queries.shape, (50, 3))
        self.assertTrue((queries < 0).any())
        self.assertTrue((queries != np.trunc(queries)).any())
