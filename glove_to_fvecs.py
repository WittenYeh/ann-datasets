#!/usr/bin/env python3
"""
Convert GloVe .txt text format to fvecs binary format.

GloVe format: each line is "word val1 val2 ... valN" with no header line.

Usage:
  python glove_to_fvecs.py <input.txt> <output.fvecs>
"""

import argparse
import sys
import os
import numpy as np


WRITE_CHUNK = 100_000


def convert_glove_to_fvecs(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.", file=sys.stderr)
        return False

    print(f"Reading {input_file}...")
    vectors_written = 0
    dim = None
    buf = None
    buf_idx = 0

    with open(input_file, "r", encoding="utf-8", errors="replace") as fin, \
         open(output_file, "wb") as fout:

        for line in fin:
            parts = line.split()
            if len(parts) < 3:
                continue

            # Detect dimension from first valid line
            if dim is None:
                dim = len(parts) - 1
                buf = np.empty((WRITE_CHUNK, dim), dtype=np.float32)
                print(f"  Detected dimension: {dim}")

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
                print(f"  Progress: {vectors_written:,}")

        if buf_idx > 0:
            _flush(fout, buf, buf_idx, dim)
            vectors_written += buf_idx

    print(f"  -> {output_file}: {vectors_written:,} vectors, dim={dim}")
    return True


def _flush(fout, buf, count, dim):
    chunk = buf[:count]
    block = np.empty((count, dim + 1), dtype="float32")
    block[:, 0] = np.array([dim], dtype="int32").view("float32")
    block[:, 1:] = chunk
    block.tofile(fout)


def main():
    parser = argparse.ArgumentParser(
        description="Convert GloVe .txt text format to fvecs binary format",
    )
    parser.add_argument("input_file", help="Input GloVe .txt file")
    parser.add_argument("output_file", help="Output .fvecs binary file")
    args = parser.parse_args()
    success = convert_glove_to_fvecs(args.input_file, args.output_file)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
