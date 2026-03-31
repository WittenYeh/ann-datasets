#!/usr/bin/env python3
"""
Extract subset from large vector datasets (fvecs, bvecs, ivecs).

Uses memory-mapped I/O to avoid loading the entire source file into RAM.

Usage:
  python extract_subset.py <input_file> <output_file> <num_vectors>

Example:
  python extract_subset.py bigann_base.bvecs sift10m_base.bvecs 10000000
"""

import argparse
import sys
import os
import numpy as np
from vecs_io import mmap_by_ext, write_by_ext


def format_size(size_bytes):
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def extract_subset(input_file, output_file, num_vectors):
    """Extract first N vectors via memory-mapped slice + vectorized write."""
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.", file=sys.stderr)
        return False

    print(f"Memory-mapping {input_file}...")
    data = mmap_by_ext(input_file)
    n, d = data.shape
    print(f"  Source: {n:,} vectors, {d} dimensions")

    actual = min(num_vectors, n)
    if actual < num_vectors:
        print(f"  Warning: only {n:,} vectors available, extracting all")

    print(f"Extracting first {actual:,} vectors...")
    subset = np.array(data[:actual])  # copy slice into RAM

    # Determine output dtype: bvecs stays uint8, fvecs/ivecs stay as-is
    write_by_ext(output_file, subset)

    output_size = os.path.getsize(output_file)
    print(f"Done. {actual:,} vectors written to {output_file} ({format_size(output_size)})")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Extract subset from large vector datasets",
        epilog="Example: python extract_subset.py bigann_base.bvecs sift10m_base.bvecs 10000000",
    )
    parser.add_argument("input_file", help="Input vector file (fvecs/bvecs/ivecs)")
    parser.add_argument("output_file", help="Output vector file")
    parser.add_argument("num_vectors", type=int, help="Number of vectors to extract")

    args = parser.parse_args()

    if args.num_vectors <= 0:
        print("Error: num_vectors must be positive", file=sys.stderr)
        sys.exit(1)

    success = extract_subset(args.input_file, args.output_file, args.num_vectors)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
