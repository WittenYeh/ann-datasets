#!/usr/bin/env python3
## @file deduplicate.py
# @brief Stable, exact fvecs/bvecs deduplication using Polars.
# @details Validate records with NumPy, find first occurrences with Polars,
# and copy retained records in parallel with Numba, preserving original order.
"""Deduplicate fvecs/bvecs with Polars and parallel output copying.

Keep the first occurrence, preserve its bytes, and assign contiguous IDs in
original order. Float equality is exact (including +0 == -0); NaN/Inf are rejected.
NumPy handles memory mappings and validation; Polars identifies unique rows;
Numba compiles parallel output copying. Requires numpy, polars and numba.
"""

import argparse
import json
import mmap
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import polars as pl
from numba import get_num_threads, njit, prange, set_num_threads


## @brief Determine the default number of worker threads.
# @return Number of CPUs allowed by process affinity, or the system CPU count
# when affinity information is unavailable; always at least one.
def default_threads():
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


## @brief Validate dimension headers and finite coordinates in bounded chunks.
# @param[in] records Complete records including dimension headers: uint32 words
# for fvecs, or uint8 bytes for bvecs; shape is (n, dim + 1) or (n, dim + 4).
# @param[in] dim Expected number of coordinates in every record.
# @param[in] is_float True for float32 fvecs; false for uint8 bvecs.
# @param[in] chunk_size Maximum records per validation chunk.
# @exception ValueError If any header is inconsistent or a float is NaN/Inf.
def _validate_records(records, dim, is_float, chunk_size=65536):
    for start in range(0, len(records), chunk_size):
        part = records[start:start + chunk_size]
        # Copy only the four header bytes for possibly unaligned bvecs records.
        headers = part[:, 0] if is_float else part[:, :4].copy().view("<u4").ravel()
        # Bound the temporary finite-value mask instead of allocating one for N x D.
        if not np.all(headers == dim) or (is_float and not np.isfinite(part[:, 1:].view("<f4")).all()):
            raise ValueError("Inconsistent dimension headers or non-finite coordinates")


## @brief Let Polars find the first occurrence of each distinct vector.
# @param[in] vectors Coordinate-only array, shape (n, dim), with finite float32
# values or uint8 values.
# @return One-dimensional array of retained original IDs, in ascending order.
# @note Each Series element is a complete vector; signed zeros compare equal.
def _unique_ids(vectors):
    ids = pl.Series("vector", vectors).arg_unique().to_numpy()
    return np.sort(ids)


## @brief Copy selected complete records into disjoint output rows in parallel.
# @param[in] records Input records including headers, viewed as uint32 or uint8.
# @param[in] ids Retained original row IDs, in the desired output order.
# @param[out] output Writable array with len(ids) rows and the same record layout.
# @note Copying integer words/bytes preserves the representative's exact float bits.
@njit(parallel=True, cache=True)
def _copy_selected(records, ids, output):
    # Copy headers and original coordinate bits, including signed zeros.
    for row in prange(len(ids)):
        for col in range(records.shape[1]):
            output[row, col] = records[ids[row], col]


## @brief Write retained records through a temporary mmap and atomically publish.
# @param[in] destination Output Path; its parent directory must already exist.
# @param[in] records Complete input records, including dimension headers.
# @param[in] ids Nonempty array of retained original row IDs.
# @return Tuple (copy_seconds, flush_seconds); excludes file setup and rename.
# @exception OSError If temporary-file creation, mapping, flushing or publication fails.
# @note The previous destination survives failures before os.replace().
def _write_output(destination, records, ids):
    # A temporary file in the same directory allows an atomic final rename.
    fd, name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    try:
        size = len(ids) * records.shape[1] * records.dtype.itemsize
        # Set the exact file length before exposing its pages as a writable array.
        os.ftruncate(fd, size)
        with mmap.mmap(fd, size, access=mmap.ACCESS_WRITE) as mapped:
            output = np.ndarray((len(ids), records.shape[1]), dtype=records.dtype, buffer=mapped)
            started = time.perf_counter()
            try:
                _copy_selected(records, ids, output)
            finally:
                # Release our array view before closing the underlying mapping.
                del output
            copied = time.perf_counter()
            # Measure filesystem flushing separately from parallel memory copying.
            mapped.flush()
            flushed = time.perf_counter()
        os.close(fd)
        fd = None
        # Publish only after the entire mapped output has been flushed and closed.
        os.replace(name, destination)
        return copied - started, flushed - copied
    finally:
        # Failed writes must not leave a partial output at the requested path.
        if fd is not None:
            os.close(fd)
        Path(name).unlink(missing_ok=True)


## @brief Publish a JSON report using a temporary file and atomic rename.
# @param[in] path Report path; its parent directory must already exist.
# @param[in] value JSON-serializable report contents.
# @exception OSError If writing or replacing the report fails.
# @exception TypeError If value contains an unsupported JSON value.
def save_json_report(path, value):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
        # Readers see either the old complete report or the new complete report.
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


## @brief Run stable exact vector deduplication and return counts and timings.
# @param[in] input_file Source .fvecs or .bvecs path; input must be nonempty.
# @param[in] output_file Destination in the same format; may equal input_file.
# @param[in] threads Parallel output-copy workers, or None for default_threads().
# Polars uses its process-wide pool, configured at startup via POLARS_MAX_THREADS.
# @param[in] report Optional JSON report path, distinct from both vector paths.
# @return Dictionary with input/unique/removed counts, dimensions, thread count,
# stage timings, paths, format and equality/representative metadata.
# @exception ValueError If arguments, records, coordinates or row counts are invalid,
# or threads exceeds Numba's configured worker limit.
# @exception OSError If input/output or report file operations fail.
# @note Keep the first occurrence's bytes, treat signed zeros as equal, and reject
# NaN/Inf. Output and report are published independently. Stage timings include
# JIT compilation or cache loading when it occurs.
def deduplicate(input_file, output_file, threads=None, report=None):
    source, destination = Path(input_file), Path(output_file)
    # Validate paths/options before mapping data or creating an output file.
    fmt = source.suffix.lstrip(".").lower()
    if fmt not in ("fvecs", "bvecs") or destination.suffix.lower() != source.suffix.lower():
        raise ValueError("Input and output must have the same .fvecs or .bvecs extension")
    threads = default_threads() if threads is None else threads
    if threads <= 0:
        raise ValueError("threads must be positive")
    if report and Path(report).resolve() in (source.resolve(), destination.resolve()):
        raise ValueError("Report path must differ from the vector paths")
    # Restore the caller's Numba thread setting even when processing raises.
    previous_threads = get_num_threads()
    set_num_threads(threads)
    started = time.perf_counter()
    try:
        with source.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            if size < 4:
                raise ValueError("Empty input or truncated dimension header")
            dim = int.from_bytes(stream.read(4), byteorder="little", signed=True)
            if dim <= 0:
                raise ValueError("Invalid dimension")
            is_float = fmt == "fvecs"
            # Each record has a four-byte dimension header followed by coordinates.
            stride = 4 + dim * (4 if is_float else 1)
            if size % stride:
                raise ValueError("Truncated vector record")
            n = size // stride
            # Keep IDs compatible with the benchmark's signed int32 ground truth.
            if n > np.iinfo(np.int32).max:
                raise ValueError("Vector count exceeds signed 32-bit IDs")
            with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                # Two zero-copy views serve byte-preserving I/O and numeric equality.
                dtype = np.dtype("<u4") if is_float else np.dtype("u1")
                records = np.ndarray((n, stride // dtype.itemsize), dtype=dtype, buffer=mapped)
                vectors = records[:, 1:].view("<f4") if is_float else records[:, 4:]
                print(f"Validating {n:,} vectors...", file=sys.stderr)
                _validate_records(records, dim, is_float)
                validated = time.perf_counter()
                print(f"Deduplicating with Polars ({pl.thread_pool_size()} thread pool)...", file=sys.stderr)
                ids = _unique_ids(vectors)
                checked = time.perf_counter()
                print(f"Writing {len(ids):,} unique vectors in original order...", file=sys.stderr)
                copy_seconds, flush_seconds = _write_output(destination, records, ids)
                del vectors, records
        done = time.perf_counter()
    finally:
        set_num_threads(previous_threads)
    # Preserve total stage timings while exposing copy/flush costs separately.
    stats = dict(input_vectors=n, unique_vectors=len(ids), removed_vectors=n - len(ids),
                 dimensions=dim, threads=threads, polars_threads=pl.thread_pool_size(),
                 validation_seconds=validated - started, deduplicate_seconds=checked - validated,
                 write_seconds=done - checked,
                 copy_seconds=copy_seconds, flush_seconds=flush_seconds,
                 total_seconds=done - started, backend="polars")
    stats.update(input=str(source.resolve()), output=str(destination.resolve()),
                 format=fmt, equality="exact coordinates; +0 equals -0", representative="first original ID")
    if report:
        save_json_report(report, stats)
    return stats


## @brief Parse CLI arguments, run deduplication and print its JSON statistics.
# @details Progress messages go to stderr; successful statistics go to stdout.
# Expected input/I/O/runtime errors produce a concise message and exit status 1.
# @exception SystemExit On argument parsing errors or an explicitly reported failure.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Original base (.fvecs or .bvecs)")
    parser.add_argument("output", help="Deduplicated base, with the same format")
    parser.add_argument("--threads", type=int, default=default_threads(),
                        help="Output-copy workers; set POLARS_MAX_THREADS before launch for Polars")
    parser.add_argument("--report", help="Write counts and timings as JSON")
    args = parser.parse_args()
    try:
        stats = deduplicate(args.input, args.output, args.threads, args.report)
    except (OSError, ValueError, RuntimeError) as error:
        # Keep expected command-line failures readable without a Python traceback.
        parser.exit(1, f"deduplicate: {error}\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
