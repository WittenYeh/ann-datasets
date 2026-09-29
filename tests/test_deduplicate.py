"""Run with: python3 -m unittest discover -s tests -v.

Install ../requirements.txt first; FAISS is required, never silently skipped.
Make integration tests also require Make >= 4.3, unzip and numactl.
"""
import io
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from unittest.mock import patch

import numpy as np
import polars as pl
from numba import get_num_threads
from numba.core.config import NUMBA_NUM_THREADS

try:
    import faiss
except ImportError as error:
    raise ImportError("Full correctness tests require faiss-cpu; run "
                      "python3 -m pip install -r requirements.txt") from error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import deduplicate as dedup_module
from deduplicate import deduplicate
import compute_groundtruth as gt_module


def fvecs(rows):
    return b"".join(struct.pack("<i", len(row)) + struct.pack("<" + "f" * len(row), *row) for row in rows)


def bvecs(rows):
    return b"".join(struct.pack("<i", len(row)) + bytes(row) for row in rows)


def reference_unique(data, fmt="fvecs"):
    """Independent numeric equality, retaining original record bytes and order."""
    dim, = struct.unpack_from("<i", data)
    stride = 4 + dim * (4 if fmt == "fvecs" else 1)
    seen, ids, records = set(), [], []
    for index, offset in enumerate(range(0, len(data), stride)):
        record = data[offset:offset + stride]
        values = struct.unpack("<" + ("f" if fmt == "fvecs" else "B") * dim, record[4:])
        if values not in seen:
            seen.add(values)
            ids.append(index)
            records.append(record)
    return b"".join(records), ids


def check_groundtruth(test, path, base, query, k):
    """Check structure and exact ranks against float64 exhaustive distances.

    Fixtures use small, well-separated coordinates (or intentional exact ties),
    so no tolerance is needed. Equal-distance IDs may appear in either order.
    """
    data = path.read_bytes()
    test.assertEqual(len(data), len(query) * (k + 1) * 4)
    gt = np.frombuffer(data, dtype="<i4").reshape(len(query), k + 1)
    test.assertTrue(np.all(gt[:, 0] == k))
    ids = gt[:, 1:]
    test.assertTrue(np.all((ids >= 0) & (ids < len(base))))
    for row in ids:
        test.assertEqual(len(set(row.tolist())), k, "GT must contain distinct IDs")
    distances = np.sum((np.asarray(query, dtype=np.float64)[:, None, :] -
                        np.asarray(base, dtype=np.float64)[None, :, :]) ** 2, axis=2)
    actual = np.take_along_axis(distances, ids, axis=1)
    np.testing.assert_array_equal(actual, np.sort(distances, axis=1)[:, :k])
    return ids


class DeduplicationTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "input.fvecs"
        self.output = self.root / "output.fvecs"

    def test_stable_parallel_exact_duplicates(self):
        # Distant duplicates exercise first-ID selection and ordered parallel copying.
        rows = [(float(i % 211), float(i % 97)) for i in range(25000)]
        expected = list(dict.fromkeys(rows))
        self.source.write_bytes(fvecs(rows))
        for threads in (1, 4):
            with self.subTest(threads=threads):
                stats = deduplicate(self.source, self.output, threads)
                self.assertEqual(self.output.read_bytes(), fvecs(expected))
                self.assertEqual(stats["unique_vectors"], len(expected))
                self.assertEqual(stats["backend"], "polars")
                self.assertEqual(stats["polars_threads"], pl.thread_pool_size())

    def test_polars_compares_complete_rows_and_preserves_readonly_input(self):
        for dtype in ("<f4", "u1"):
            for dim in (1, 3, 128, 960):
                with self.subTest(dtype=dtype, dim=dim):
                    records = np.zeros((5, dim + 1), dtype=dtype)
                    vectors = records[:, 1:]  # Non-contiguous rows, like vecs files.
                    vectors[:] = np.arange(dim) % 100
                    vectors[2, -1] += 1
                    vectors[4] = vectors[2]
                    vectors[4, 0] += 10
                    if dtype == "<f4":
                        vectors[3, 0] = -0.0
                    snapshot = records.tobytes()
                    vectors.flags.writeable = False
                    np.testing.assert_array_equal(dedup_module._unique_ids(vectors), [0, 2, 4])
                    self.assertEqual(records.tobytes(), snapshot)

    def test_cli_reports_polars_and_copy_thread_settings(self):
        rows = [(-0.0, 1.0), (0.0, 1.0), (2.0, 3.0), (2.0, 3.0)]
        self.source.write_bytes(fvecs(rows))
        env = dict(os.environ, POLARS_MAX_THREADS="3")
        command = [sys.executable, str(ROOT / "deduplicate.py"), str(self.source),
                   str(self.output), "--threads", "2"]
        result = subprocess.run(command, env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        stats = json.loads(result.stdout)
        self.assertEqual(stats["backend"], "polars")
        self.assertEqual(stats["threads"], 2)
        self.assertEqual(stats["polars_threads"], 3)
        self.assertGreaterEqual(stats["validation_seconds"], 0)
        self.assertGreaterEqual(stats["deduplicate_seconds"], 0)
        self.assertEqual(self.output.read_bytes(), fvecs([rows[0], rows[2]]))

    def test_signed_zero_keeps_first_bytes(self):
        rows = [(-0.0, 1.0), (0.0, 1.0), (0.0, 2.0), (-0.0, 2.0)]
        self.source.write_bytes(fvecs(rows))
        deduplicate(self.source, self.output, 4)
        self.assertEqual(self.output.read_bytes(), fvecs([rows[0], rows[2]]))

    def test_one_ulp_difference_at_one_is_distinct(self):
        self.source.write_bytes(struct.pack("<IIII", 1, 0x3f800000, 1, 0x3f800001))
        deduplicate(self.source, self.output, 2)
        self.assertEqual(self.output.read_bytes(), self.source.read_bytes())

    def test_all_identical_and_in_place(self):
        self.source.write_bytes(fvecs([(1.0, 2.0)] * 100))
        stats = deduplicate(self.source, self.source, 2, self.root / "report.json")
        self.assertEqual(self.source.read_bytes(), fvecs([(1.0, 2.0)]))
        self.assertEqual(stats["removed_vectors"], 99)
        self.assertEqual(json.loads((self.root / "report.json").read_text()), stats)

    def test_bvecs_unaligned_records(self):
        self.source = self.root / "input.bvecs"
        self.output = self.root / "output.bvecs"
        records = [struct.pack("<iBBB", 3, *v) for v in [(1, 2, 3), (2, 3, 4), (1, 2, 3)]]
        self.source.write_bytes(b"".join(records))
        deduplicate(self.source, self.output, 4)
        self.assertEqual(self.output.read_bytes(), b"".join(records[:2]))

    def test_malformed_input_preserves_previous_output(self):
        invalid = [b"", b"\x01", struct.pack("<i", 0), struct.pack("<i", -1), fvecs([(1.0, 2.0)])[:-1],
                   fvecs([(1.0, 2.0)]) + struct.pack("<iff", 3, 2.0, 3.0),
                   fvecs([(float("nan"),)]), fvecs([(float("inf"),)])]
        self.output.write_bytes(b"previous output")
        for data in invalid:
            with self.subTest(data=data):
                self.source.write_bytes(data)
                with self.assertRaises(ValueError):
                    deduplicate(self.source, self.output, 2)
                self.assertEqual(self.output.read_bytes(), b"previous output")
                self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_invalid_record_after_first_validation_chunk_preserves_output(self):
        records = np.ones((65537, 2), dtype="<u4")
        self.output.write_bytes(b"previous output")
        for header, coordinate in ((2, 0), (1, 0x7fc00000), (1, 0xff800000)):
            with self.subTest(header=header, coordinate=coordinate):
                records[-1] = header, coordinate
                self.source.write_bytes(records.tobytes())
                with self.assertRaises(ValueError):
                    deduplicate(self.source, self.output, 2)
                self.assertEqual(self.output.read_bytes(), b"previous output")

    def test_invalid_bvecs_header_preserves_output(self):
        source, output = self.root / "input.bvecs", self.root / "output.bvecs"
        source.write_bytes(struct.pack("<iBBB", 3, 1, 2, 3) + struct.pack("<iBBB", 4, 1, 2, 3))
        output.write_bytes(b"previous output")
        with self.assertRaises(ValueError):
            deduplicate(source, output, 4)
        self.assertEqual(output.read_bytes(), b"previous output")

    def test_interrupted_copy_preserves_output_and_thread_count(self):
        self.source.write_bytes(fvecs([(1.0, 2.0), (3.0, 4.0)]))
        self.output.write_bytes(b"previous output")
        previous_threads = get_num_threads()

        def fail_after_partial_copy(records, ids, output):
            output[0] = records[ids[0]]
            raise OSError("Simulated write failure")

        with patch.object(dedup_module, "_copy_selected", side_effect=fail_after_partial_copy):
            with self.assertRaisesRegex(OSError, "Simulated write failure"):
                deduplicate(self.source, self.output, 2)
        self.assertEqual(get_num_threads(), previous_threads)
        self.assertEqual(self.output.read_bytes(), b"previous output")
        self.assertEqual(list(self.root.glob("*.tmp")), [])


    def test_randomized_rows_match_independent_reference(self):
        rng = np.random.default_rng(20260926)
        for fmt, writer in (("fvecs", fvecs), ("bvecs", bvecs)):
            source, output = self.root / ("input." + fmt), self.root / ("output." + fmt)
            for dim in (1, 3, 128, 960):
                # Includes identical, distinct and partially repeated input sets.
                pool = (rng.standard_normal((129, dim)).astype("<f4") if fmt == "fvecs" else
                        rng.integers(0, 256, size=(129, dim), dtype=np.uint8))
                for indices in (np.zeros(1, dtype=int), np.zeros(129, dtype=int), np.arange(129),
                                rng.integers(0, 129, size=513)):
                    data = writer(pool[indices].tolist())
                    expected, ids = reference_unique(data, fmt)
                    source.write_bytes(data)
                    for threads in (1, 4):
                        with self.subTest(fmt=fmt, dim=dim, rows=len(indices), threads=threads):
                            previous_threads = get_num_threads()
                            stats = deduplicate(source, output, threads)
                            self.assertEqual(output.read_bytes(), expected)
                            self.assertEqual(source.read_bytes(), data)
                            self.assertEqual(get_num_threads(), previous_threads)
                            self.assertEqual(stats["input_vectors"], len(indices))
                            self.assertEqual(stats["unique_vectors"], len(ids))
                            self.assertEqual(stats["removed_vectors"], len(indices) - len(ids))
                            self.assertEqual(stats["dimensions"], dim)
                    # A second deduplication must preserve every byte.
                    deduplicate(output, output, 2)
                    self.assertEqual(output.read_bytes(), expected)

    def test_every_coordinate_can_distinguish_vectors(self):
        for dtype in ("<f4", "u1"):
            for dim in (1, 3, 128, 960):
                with self.subTest(dtype=dtype, dim=dim):
                    rows = np.vstack([np.zeros((1, dim), dtype=dtype), np.eye(dim, dtype=dtype)])
                    rows = np.vstack([rows, rows[::-1]])
                    np.testing.assert_array_equal(dedup_module._unique_ids(rows), np.arange(dim + 1))

    def test_float32_extremes_subnormals_and_signed_zeros(self):
        bits = [0x80000000, 0, 1, 2, 0x80000001, 0x80000002,
                0x007fffff, 0x00800000, 0x3f800000, 0x3f800001,
                0x7f7fffff, 0xff7fffff]
        data = b"".join(struct.pack("<II", 1, word) for word in bits + bits[::-1])
        expected, _ = reference_unique(data)
        self.source.write_bytes(data)
        deduplicate(self.source, self.output, 4)
        self.assertEqual(self.output.read_bytes(), expected)

    def test_polars_thread_counts_produce_identical_output(self):
        rng = np.random.default_rng(731)
        pool = rng.integers(-100, 100, size=(1024, 9)).astype("<f4")
        pool[0, 0] = -0.0
        rows = np.vstack([pool, pool[rng.permutation(len(pool))], pool[:100]])
        data = fvecs(rows.tolist())
        expected, _ = reference_unique(data)
        self.source.write_bytes(data)
        for threads in (1, 4):
            with self.subTest(polars_threads=threads):
                result = subprocess.run(
                    [sys.executable, str(ROOT / "deduplicate.py"), str(self.source),
                     str(self.output), "--threads", "2"],
                    env=dict(os.environ, POLARS_MAX_THREADS=str(threads)),
                    text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)["polars_threads"], threads)
                self.assertEqual(self.output.read_bytes(), expected)

    def test_invalid_options_preserve_files_and_thread_count(self):
        data = fvecs([(1.0, 2.0)])
        for options in ({"threads": 0}, {"threads": -1}, {"threads": NUMBA_NUM_THREADS + 1},
                        {"output_file": self.root / "output.bvecs"},
                        {"input_file": self.root / "input.bin"},
                        {"report": self.source}, {"report": self.output}):
            args = dict(input_file=self.source, output_file=self.output, threads=2)
            args.update(options)
            with self.subTest(options=options):
                args["input_file"].write_bytes(data)
                args["output_file"].write_bytes(b"previous output")
                previous_threads = get_num_threads()
                with self.assertRaises(ValueError):
                    deduplicate(**args)
                self.assertEqual(args["input_file"].read_bytes(), data)
                self.assertEqual(args["output_file"].read_bytes(), b"previous output")
                self.assertEqual(get_num_threads(), previous_threads)
                self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_output_io_failures_preserve_files(self):
        class FailingFlush(dedup_module.mmap.mmap):
            def flush(self, *args):
                raise OSError("Simulated flush failure")

        data = fvecs([(1.0, 2.0), (1.0, 2.0)])
        for stage in ("truncate", "flush", "replace"):
            for mode in ("existing", "new", "in-place"):
                with self.subTest(stage=stage, mode=mode):
                    self.source.write_bytes(data)
                    self.output.unlink(missing_ok=True)
                    if mode == "existing":
                        self.output.write_bytes(b"previous output")
                    destination = self.source if mode == "in-place" else self.output
                    if stage == "flush":
                        failure = patch.object(dedup_module.mmap, "mmap", FailingFlush)
                    else:
                        name = "ftruncate" if stage == "truncate" else "replace"
                        failure = patch.object(dedup_module.os, name, side_effect=OSError("Simulated I/O failure"))
                    previous_threads = get_num_threads()
                    with failure, self.assertRaisesRegex(OSError, "Simulated"):
                        deduplicate(self.source, destination, 2)
                    self.assertEqual(self.source.read_bytes(), data)
                    if mode == "existing":
                        self.assertEqual(self.output.read_bytes(), b"previous output")
                    elif mode == "new":
                        self.assertFalse(self.output.exists())
                    self.assertEqual(get_num_threads(), previous_threads)
                    self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_json_report_failure_preserves_previous_report(self):
        report = self.root / "report.json"
        report.write_bytes(b"previous report")
        with self.assertRaises(TypeError):
            dedup_module.save_json_report(report, {"not_json_serializable": {1, 2}})
        self.assertEqual(report.read_bytes(), b"previous report")
        self.assertEqual(list(self.root.glob("*.tmp")), [])
        with patch.object(dedup_module.os, "replace", side_effect=OSError("Report publish failure")):
            with self.assertRaises(OSError):
                dedup_module.save_json_report(report, {"ok": True})
        self.assertEqual(report.read_bytes(), b"previous report")
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_report_failure_after_output_publication(self):
        # Data and report are separate atomic publications, not one transaction.
        data = fvecs([(2.0, 1.0), (2.0, 1.0), (0.0, 1.0)])
        self.source.write_bytes(data)
        report = self.root / "report.json"
        report.write_bytes(b"previous report")
        with patch.object(dedup_module, "save_json_report", side_effect=OSError("Report failure")):
            with self.assertRaises(OSError):
                deduplicate(self.source, self.output, 2, report)
        self.assertEqual(self.output.read_bytes(), reference_unique(data)[0])
        self.assertEqual(report.read_bytes(), b"previous report")
        self.assertEqual(list(self.root.glob("*.tmp")), [])


class GroundTruthTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.output = self.root / "gt.ivecs"

    def test_single_and_chunked_search_match_exhaustive_reference(self):
        rng = np.random.default_rng(421)
        base = rng.integers(0, 32, size=(37, 5))
        query = np.vstack([base[0], rng.integers(0, 32, size=(2, 5))])
        for fmt, writer in (("fvecs", fvecs), ("bvecs", bvecs)):
            source, queries = self.root / ("base." + fmt), self.root / ("query." + fmt)
            source.write_bytes(writer(base.tolist()))
            queries.write_bytes(writer(query.tolist()))
            for k in (1, 7, len(base)):
                # chunk=1 and short final chunks exercise FAISS -1/inf padding.
                for chunk in (1, 5, 13, len(base), 100):
                    with self.subTest(fmt=fmt, k=k, chunk=chunk):
                        gt_module.compute_groundtruth(str(source), str(queries), str(self.output), k, chunk)
                        check_groundtruth(self, self.output, base, query, k)

    def test_top1000_and_short_final_chunk(self):
        rng = np.random.default_rng(173)
        base = rng.integers(-100, 101, size=(1003, 3))
        query = np.array([[0, 0, 0], [11, -13, 17]])
        source, queries = self.root / "base.fvecs", self.root / "query.fvecs"
        source.write_bytes(fvecs(base.tolist()))
        queries.write_bytes(fvecs(query.tolist()))
        for chunk in (257, len(base)):
            with self.subTest(chunk=chunk):
                gt_module.compute_groundtruth(str(source), str(queries), str(self.output), 1000, chunk)
                check_groundtruth(self, self.output, base, query, 1000)

    def test_equal_distance_boundary_and_duplicate_id_detection(self):
        base = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [2, 0], [-2, 0]])
        query = np.array([[0, 0]])
        source, queries = self.root / "base.fvecs", self.root / "query.fvecs"
        source.write_bytes(fvecs(base.tolist()))
        queries.write_bytes(fvecs(query.tolist()))
        for k in (3, 5, 7):
            for chunk in (1, 2, 4, len(base)):
                with self.subTest(k=k, chunk=chunk):
                    gt_module.compute_groundtruth(str(source), str(queries), str(self.output), k, chunk)
                    check_groundtruth(self, self.output, base, query, k)
        # Any distinct IDs at the tied boundary are valid; repeating one is not.
        self.output.write_bytes(struct.pack("<iiii", 3, 0, 3, 4))
        check_groundtruth(self, self.output, base, query, 3)
        self.output.write_bytes(struct.pack("<iiii", 3, 0, 1, 1))
        with self.assertRaisesRegex(AssertionError, "distinct IDs"):
            check_groundtruth(self, self.output, base, query, 3)

    def test_invalid_gt_files_are_rejected_by_checker(self):
        base, query = [[0, 0], [1, 0], [3, 0]], [[0, 0]]
        for data in (struct.pack("<iii", 1, 0, 1),   # Wrong row header.
                     struct.pack("<iii", 2, -1, 0),  # Negative ID.
                     struct.pack("<iii", 2, 0, 3),   # Out-of-range ID.
                     struct.pack("<iii", 2, 1, 0),   # Wrong ordering.
                     struct.pack("<iii", 2, 0, 2),   # Missing a closer ID.
                     struct.pack("<ii", 2, 0)):      # Truncated row.
            with self.subTest(data=data):
                self.output.write_bytes(data)
                with self.assertRaises(AssertionError):
                    check_groundtruth(self, self.output, base, query, 2)

    def test_dimension_mismatch_preserves_previous_gt(self):
        source, query = self.root / "base.fvecs", self.root / "query.fvecs"
        source.write_bytes(fvecs([(1, 2), (3, 4)]))
        query.write_bytes(fvecs([(1,)]))
        self.output.write_bytes(b"previous GT")
        with self.assertRaisesRegex(ValueError, "Dimension mismatch"):
            gt_module.compute_groundtruth(str(source), str(query), str(self.output), 1, 10)
        self.assertEqual(self.output.read_bytes(), b"previous GT")
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_write_and_publish_failures_preserve_previous_gt(self):
        source, query = self.root / "base.fvecs", self.root / "query.fvecs"
        source_data, query_data = fvecs([(1, 2), (3, 4)]), fvecs([(1, 1)])
        source.write_bytes(source_data)
        query.write_bytes(query_data)

        def partial_write(path, ids):
            Path(path).write_bytes(b"partial GT")
            raise OSError("Simulated GT write failure")

        for stage in ("write", "replace"):
            for exists in (False, True):
                with self.subTest(stage=stage, previous_output=exists):
                    self.output.unlink(missing_ok=True)
                    if exists:
                        self.output.write_bytes(b"previous GT")
                    failure = (patch.object(gt_module, "ivecs_write", side_effect=partial_write)
                               if stage == "write" else
                               patch.object(gt_module.os, "replace", side_effect=OSError("Simulated GT publish failure")))
                    with failure, self.assertRaisesRegex(OSError, "Simulated GT"):
                        gt_module.compute_groundtruth(str(source), str(query), str(self.output), 1, 10)
                    if exists:
                        self.assertEqual(self.output.read_bytes(), b"previous GT")
                    else:
                        self.assertFalse(self.output.exists())
                    self.assertEqual(source.read_bytes(), source_data)
                    self.assertEqual(query.read_bytes(), query_data)
                    self.assertEqual(list(self.root.glob("*.tmp")), [])


class MakePipelineTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for p in list(ROOT.glob("*.py")) + list(ROOT.glob("*.mk")):
            (self.root / p.name).symlink_to(p)
        # Unsorted first occurrences expose accidental sorting/ID remapping.
        values = [float((7 * i + 3) % 20) for i in range(60)]
        self.rows = [(v, v * v) for v in values]
        self.queries = [(0.1, 0.2), (8.3, 61.7), (13.1, 170.2)]
        self.base_bytes, self.query_bytes = fvecs(self.rows), fvecs(self.queries)
        # Fixed expected positions for the documented seed=42, n=60, nq=3.
        query_ids = (5, 39, 45)
        self.split_base = fvecs([row for i, row in enumerate(self.rows) if i not in query_ids])
        self.split_query = fvecs([self.rows[i] for i in query_ids])

    def archive(self, directory, archive_name, prefix, members):
        with tarfile.open(directory / archive_name, "w:gz") as tar:
            for name, data in members.items():
                info = tarfile.TarInfo(prefix + "/" + name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))

    def make(self, directory, *extra):
        command = ["make", "--no-print-directory", "-j4", "all", "GT_K=7", "GT_CHUNK_SIZE=13",
                   "NUM_QUERIES=3", "NUM_VECTORS=60", *extra]
        result = subprocess.run(command, cwd=directory, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        self.assertEqual(result.returncode, 0, result.stdout)
        return result.stdout

    def check(self, directory, prefix, expected_raw, expected_query, k=7):
        self.assertEqual((directory / (prefix + "_base.raw.fvecs")).read_bytes(), expected_raw)
        expected_base, ids = reference_unique(expected_raw)
        self.assertEqual((directory / (prefix + "_base.fvecs")).read_bytes(), expected_base,
                         "Base differs from the complete stable unique input")
        query_path = directory / (prefix + "_query.fvecs")
        self.assertEqual(query_path.read_bytes(), expected_query)
        dim, = struct.unpack_from("<i", expected_raw)
        # Decode independently of production vecs_io and compare against input fixtures.
        base = np.frombuffer(expected_base, dtype="<f4").reshape(-1, dim + 1)[:, 1:]
        query = np.frombuffer(expected_query, dtype="<f4").reshape(-1, dim + 1)[:, 1:]
        stats = json.loads((directory / (prefix + "_base.fvecs.dedup.json")).read_text())
        self.assertEqual(stats["backend"], "polars")
        self.assertEqual(stats["polars_threads"], int(subprocess.check_output(["nproc"], text=True)))
        count = len(expected_raw) // ((dim + 1) * 4)
        self.assertEqual(stats["input_vectors"], count)
        self.assertEqual(stats["unique_vectors"], len(ids))
        self.assertEqual(stats["removed_vectors"], count - len(ids))
        self.assertEqual(stats["dimensions"], dim)
        check_groundtruth(self, directory / (prefix + "_groundtruth.ivecs"), base, query, k)

    def test_all_eight_makefiles_and_incremental_builds(self):
        jobs = [("sift-1m", "sift"), ("gist-1m", "gist"), ("tiny5m", "tiny5m"),
                ("crawl", "crawl"), ("glove-100d", "glove100d"), ("yahoomusic", "yahoomusic"),
                ("deep-10m", "deep10m"), ("spacev-10m", "spacev10m")]
        for name, prefix in jobs:
            with self.subTest(dataset=name):
                directory = self.root / name
                directory.mkdir()
                shutil.copyfile(ROOT / name / "Makefile", directory / "Makefile")
                expected_raw = self.base_bytes
                expected_query = self.query_bytes
                if name in ("sift-1m", "gist-1m", "tiny5m"):
                    members = {prefix + "_base.fvecs": self.base_bytes,
                               prefix + "_query.fvecs": self.query_bytes,
                               prefix + "_learn.fvecs": self.base_bytes,
                               prefix + "_groundtruth.ivecs": b"INVALID OFFICIAL GT"}
                    self.archive(directory, prefix + ".tar.gz", prefix, members)
                elif name == "yahoomusic":
                    self.archive(directory, "yahoomusic.tar.gz", name,
                                 {"yahoomusic_query_all.fvecs": self.base_bytes})
                    expected_raw, expected_query = self.split_base, self.split_query
                elif name in ("crawl", "glove-100d"):
                    archive, member = (("crawl-300d-2M.vec.zip", "crawl-300d-2M.vec") if name == "crawl" else
                                       ("glove.2024.wikigiga.100d.zip", "glove_100_test_combined.txt"))
                    text = "".join(f"word{i} {x} {y}\n" for i, (x, y) in enumerate(self.rows))
                    if name == "crawl":
                        text = "60 2\n" + text
                    with zipfile.ZipFile(directory / archive, "w") as zipped:
                        zipped.writestr(member, text)
                    expected_raw, expected_query = self.split_base, self.split_query
                elif name == "deep-10m":
                    parent = self.root / "deep-1b"
                    parent.mkdir()
                    for suffix, data in [("base", self.base_bytes), ("queries", self.query_bytes), ("learn", self.base_bytes)]:
                        (parent / ("deep1B_" + suffix + ".fvecs")).write_bytes(data)
                else:
                    # Signed i8 conversion, with duplicates and negative values.
                    rows = [(i % 20 - 10, i % 20) for i in range(60)]
                    queries = [(1, 2), (3, 4), (5, 6)]
                    for filename, data in [("spacev1b_base.10M.i8bin", rows), ("query.i8bin", queries)]:
                        (directory / filename).write_bytes(struct.pack("<II", len(data), 2) +
                                                           b"".join(struct.pack("<bb", *r) for r in data))
                    expected_query = fvecs(queries)
                    expected_raw = fvecs(rows)
                first = self.make(directory)
                self.assertIn("../deduplicate.py", first)
                self.assertIn("../compute_groundtruth.py", first)
                self.assertNotIn("bin_to_ivecs", first)
                self.assertNotIn("Downloading", first)
                self.check(directory, prefix, expected_raw, expected_query)
                files = [directory / (prefix + suffix) for suffix in
                         ("_base.fvecs", "_query.fvecs", "_groundtruth.ivecs")]
                mtimes = [p.stat().st_mtime_ns for p in files]
                self.make(directory)
                self.assertEqual(mtimes, [p.stat().st_mtime_ns for p in files])
                changed = self.make(directory, "GT_K=4")
                self.assertNotIn("../deduplicate.py", changed)
                self.assertIn("../compute_groundtruth.py", changed)
                self.check(directory, prefix, expected_raw, expected_query, k=4)
                changed = self.make(directory, "GT_K=4", "GT_CHUNK_SIZE=5")
                self.assertNotIn("../deduplicate.py", changed)
                self.assertIn("../compute_groundtruth.py", changed)
                self.check(directory, prefix, expected_raw, expected_query, k=4)
                if name == "sift-1m":
                    for missing in files + [directory / (prefix + "_base.fvecs.dedup.json")]:
                        missing.unlink()
                        self.make(directory)
                        self.check(directory, prefix, expected_raw, expected_query)

    def test_checker_rejects_incomplete_base_even_with_consistent_gt(self):
        prefix = "bad"
        (self.root / (prefix + "_base.raw.fvecs")).write_bytes(self.base_bytes)
        unique, _ = reference_unique(self.base_bytes)
        base = self.root / (prefix + "_base.fvecs")
        base.write_bytes(unique[:-12])  # Deliberately omit one distinct 2D vector.
        (self.root / (prefix + "_base.fvecs.dedup.json")).write_text(json.dumps({
            "backend": "polars", "polars_threads": int(subprocess.check_output(["nproc"], text=True)),
            "input_vectors": 60, "unique_vectors": 19, "removed_vectors": 41, "dimensions": 2,
        }))
        query = self.root / (prefix + "_query.fvecs")
        query.write_bytes(self.query_bytes)
        gt_module.compute_groundtruth(str(base), str(query),
                                      str(self.root / (prefix + "_groundtruth.ivecs")), 7, 13)
        with self.assertRaisesRegex(AssertionError, "complete stable unique input"):
            self.check(self.root, prefix, self.base_bytes, self.query_bytes)

    def test_gt_failure_preserves_output(self):
        from compute_groundtruth import compute_groundtruth
        source, query, gt = (self.root / n for n in ("base.fvecs", "query.fvecs", "gt.ivecs"))
        source.write_bytes(self.base_bytes)
        query.write_bytes(self.query_bytes)
        gt.write_bytes(b"previous GT")
        for k, chunk in [(1000, 13), (0, 13), (7, 0)]:
            with self.assertRaises(ValueError):
                compute_groundtruth(str(source), str(query), str(gt), k, chunk)
            self.assertEqual(gt.read_bytes(), b"previous GT")


if __name__ == "__main__":
    unittest.main()
