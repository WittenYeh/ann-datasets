#!/usr/bin/env python3
"""
Convert fastText .vec text format to fvecs binary format.

Writes output in streaming chunks to avoid holding all vectors in RAM.

Usage:
  python vec_to_fvecs.py <input.vec> <output.fvecs> [--max-vectors N]

Example:
  python vec_to_fvecs.py crawl-300d-2M.vec crawl_base.fvecs
"""

import argparse
import sys
import os
import numpy as np
from vecs_io import fvecs_write

WRITE_CHUNK = 100_000  # flush to disk every 100K vectors


def convert_vec_to_fvecs(input_file, output_file, max_vectors=None):
    """Convert fastText .vec text file to fvecs binary format."""
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.", file=sys.stderr)
        return False

    print(f"Reading {input_file}...")
    vectors_written = 0

    with open(input_file, "r", encoding="utf-8", errors="replace") as fin, \
         open(output_file, "wb") as fout:

        header = fin.readline().strip().split()
        num_vectors = int(header[0])
        dim = int(header[1])
        print(f"  Header: {num_vectors:,} vectors, dim={dim}")

        if max_vectors is not None:
            num_vectors = min(num_vectors, max_vectors)
            print(f"  Limiting to {num_vectors:,} vectors")

        buf = np.empty((WRITE_CHUNK, dim), dtype=np.float32)
        buf_idx = 0

        for line in fin:
            if vectors_written + buf_idx >= num_vectors:
                break
            parts = line.split()
            if len(parts) != dim + 1:
                continue
            try:
                buf[buf_idx] = [float(x) for x in parts[1:]]
            except ValueError:
                continue
            buf_idx += 1

            if buf_idx == WRITE_CHUNK:
                _flush(fout, buf, buf_idx, dim)
                vectors_written += buf_idx
                buf_idx = 0
                print(f"  Progress: {vectors_written:,} / {num_vectors:,}")

        # flush remainder
        if buf_idx > 0:
            _flush(fout, buf, buf_idx, dim)
            vectors_written += buf_idx

    print(f"  -> {output_file}: {vectors_written:,} vectors, dim={dim}")
    return True


def _flush(fout, buf, count, dim):
    """Write `count` rows from buf to fout in fvecs format."""
    chunk = buf[:count]
    block = np.empty((count, dim + 1), dtype="float32")
    block[:, 0] = np.array([dim], dtype="int32").view("float32")
    block[:, 1:] = chunk
    block.tofile(fout)


def main():
    parser = argparse.ArgumentParser(
        description="Convert fastText .vec text format to fvecs binary format",
        epilog="Example: python vec_to_fvecs.py crawl-300d-2M.vec crawl_base.fvecs",
    )
    parser.add_argument("input_file", help="Input .vec text file")
    parser.add_argument("output_file", help="Output .fvecs binary file")
    parser.add_argument(
        "-n", "--max-vectors", type=int, default=None,
        help="Maximum number of vectors to convert",
    )

    args = parser.parse_args()
    success = convert_vec_to_fvecs(args.input_file, args.output_file, args.max_vectors)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
