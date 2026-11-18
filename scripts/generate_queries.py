#!/usr/bin/env python3
"""Generate float32 queries by adding independent Gaussian noise to sampled base rows."""

import argparse
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from vecs_io import fvecs_write, mmap_by_ext


BATCH_SIZE = 4096


def generate_queries(base_file, query_file, num_queries, variance):
    source, destination = Path(base_file), Path(query_file)
    if source.suffix.lower() not in (".fvecs", ".bvecs") or destination.suffix.lower() != ".fvecs":
        raise ValueError("Expected a .fvecs or .bvecs base and a .fvecs query file")
    if destination.exists() and source.samefile(destination):
        raise ValueError("The query output must be a different file from the base")
    if not math.isfinite(variance) or variance < 0:
        raise ValueError("variance must be finite and nonnegative")

    base = mmap_by_ext(str(source))
    if base.shape[1] == 0:
        raise ValueError("Base vectors must have a positive dimension")
    if not 1 <= num_queries <= len(base):
        raise ValueError(f"num_queries must be between 1 and the base size ({len(base)})")

    rng = np.random.default_rng(42)
    source_ids = rng.choice(len(base), size=num_queries, replace=False)
    stddev = math.sqrt(variance)
    fd, temporary = tempfile.mkstemp(dir=destination.parent, prefix=destination.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            for start in range(0, num_queries, BATCH_SIZE):
                rows = base[source_ids[start:start + BATCH_SIZE]].astype(np.float32, copy=False)
                if variance:
                    with np.errstate(over="raise"):
                        rows += rng.normal(0.0, stddev, size=rows.shape)
                if not np.isfinite(rows).all():
                    raise ValueError("Query coordinates must be finite float32 values")
                fvecs_write(stream, rows)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_file", help="Prepared base (.fvecs or .bvecs)")
    parser.add_argument("query_file", help="Output queries (.fvecs)")
    parser.add_argument("--num-queries", "--num_queries", type=int, required=True,
                        help="Number of distinct base rows to sample")
    parser.add_argument("--variance", type=float, required=True,
                        help="Gaussian noise variance per coordinate, in squared base units")
    args = parser.parse_args()
    try:
        generate_queries(args.base_file, args.query_file, args.num_queries, args.variance)
    except (ValueError, OSError, FloatingPointError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
