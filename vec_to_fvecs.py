#!/usr/bin/env python3
"""
Convert fastText .vec text format to fvecs binary format.

fastText .vec format:
  First line: <num_vectors> <dimension>
  Subsequent lines: <word> <float1> <float2> ... <floatN>

Usage:
  python vec_to_fvecs.py <input.vec> <output.fvecs> [--max-vectors N]

Example:
  python vec_to_fvecs.py crawl-300d-2M.vec crawl_base.fvecs
"""

import argparse
import struct
import sys
import os
import numpy as np


def convert_vec_to_fvecs(input_file, output_file, max_vectors=None):
    """Convert fastText .vec text file to fvecs binary format."""
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.", file=sys.stderr)
        return False

    print(f"Reading {input_file}...")
    vectors_written = 0

    with open(input_file, 'r', encoding='utf-8', errors='replace') as fin, \
         open(output_file, 'wb') as fout:

        # Read header line
        header = fin.readline().strip().split()
        num_vectors = int(header[0])
        dim = int(header[1])
        print(f"  Header: {num_vectors:,} vectors, dim={dim}")

        if max_vectors is not None:
            num_vectors = min(num_vectors, max_vectors)
            print(f"  Limiting to {num_vectors:,} vectors")

        for line in fin:
            if vectors_written >= num_vectors:
                break

            parts = line.strip().split()
            if len(parts) != dim + 1:
                # Skip malformed lines (e.g., multi-word entries)
                continue

            try:
                vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            except ValueError:
                continue

            if len(vec) != dim:
                continue

            fout.write(struct.pack('i', dim))
            fout.write(vec.tobytes())
            vectors_written += 1

            if vectors_written % 100000 == 0:
                print(f"  Progress: {vectors_written:,} / {num_vectors:,}")

    print(f"  -> {output_file}: {vectors_written:,} vectors, dim={dim}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert fastText .vec text format to fvecs binary format",
        epilog="Example: python vec_to_fvecs.py crawl-300d-2M.vec crawl_base.fvecs"
    )
    parser.add_argument("input_file", help="Input .vec text file")
    parser.add_argument("output_file", help="Output .fvecs binary file")
    parser.add_argument("-n", "--max-vectors", type=int, default=None,
                        help="Maximum number of vectors to convert")

    args = parser.parse_args()
    success = convert_vec_to_fvecs(args.input_file, args.output_file, args.max_vectors)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
