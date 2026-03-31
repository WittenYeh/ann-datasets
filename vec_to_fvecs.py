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
import sys
import os
import numpy as np
from vecs_io import fvecs_write


def convert_vec_to_fvecs(input_file, output_file, max_vectors=None):
    """Convert fastText .vec text file to fvecs binary format."""
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.", file=sys.stderr)
        return False

    print(f"Reading {input_file}...")

    with open(input_file, "r", encoding="utf-8", errors="replace") as fin:
        header = fin.readline().strip().split()
        num_vectors = int(header[0])
        dim = int(header[1])
        print(f"  Header: {num_vectors:,} vectors, dim={dim}")

        if max_vectors is not None:
            num_vectors = min(num_vectors, max_vectors)
            print(f"  Limiting to {num_vectors:,} vectors")

        vecs = []
        for line in fin:
            if len(vecs) >= num_vectors:
                break
            parts = line.strip().split()
            if len(parts) != dim + 1:
                continue
            try:
                vec = [float(x) for x in parts[1:]]
            except ValueError:
                continue
            if len(vec) != dim:
                continue
            vecs.append(vec)
            if len(vecs) % 100000 == 0:
                print(f"  Progress: {len(vecs):,} / {num_vectors:,}")

    data = np.array(vecs, dtype=np.float32)
    print(f"Writing {output_file}...")
    fvecs_write(output_file, data)
    print(f"  -> {output_file}: {data.shape[0]:,} vectors, dim={dim}")
    return True


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
