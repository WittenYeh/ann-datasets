#!/usr/bin/env python3
"""
Extract subset from large vector datasets (fvecs, bvecs, ivecs).

Usage:
  python extract_subset.py <input_file> <output_file> <num_vectors>

Example:
  python extract_subset.py bigann_base.bvecs sift10m_base.bvecs 10000000
"""

import argparse
import struct
import sys
import os


def get_format_info(filename):
    """Determine vector format from file extension."""
    ext = filename.lower().split('.')[-1]
    if ext == 'fvecs':
        return 'f', 4
    elif ext == 'bvecs':
        return 'B', 1
    elif ext == 'ivecs':
        return 'i', 4
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def extract_subset(input_file, output_file, num_vectors):
    """Extract first N vectors from input file to output file."""

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.", file=sys.stderr)
        return False

    try:
        format_char, elem_size = get_format_info(input_file)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return False

    print(f"Extracting {num_vectors:,} vectors from {input_file}...")

    with open(input_file, 'rb') as fin, open(output_file, 'wb') as fout:
        # Read first vector to get dimension
        dim_bytes = fin.read(4)
        if len(dim_bytes) < 4:
            print("Error: Input file is empty or corrupted.", file=sys.stderr)
            return False

        dim = struct.unpack('i', dim_bytes)[0]
        vec_size_bytes = 4 + dim * elem_size

        # Reset to beginning
        fin.seek(0)

        # Copy vectors
        vectors_copied = 0
        chunk_size = 1000  # Copy 1000 vectors at a time

        while vectors_copied < num_vectors:
            vectors_to_copy = min(chunk_size, num_vectors - vectors_copied)
            bytes_to_copy = vectors_to_copy * vec_size_bytes

            data = fin.read(bytes_to_copy)
            if len(data) < bytes_to_copy:
                print(f"Warning: Only {vectors_copied} vectors available in input file.")
                break

            fout.write(data)
            vectors_copied += vectors_to_copy

            if vectors_copied % 100000 == 0:
                print(f"  Progress: {vectors_copied:,} / {num_vectors:,} vectors")

    print(f"✓ Successfully extracted {vectors_copied:,} vectors to {output_file}")

    # Show file sizes
    input_size = os.path.getsize(input_file)
    output_size = os.path.getsize(output_file)
    print(f"  Input size:  {format_size(input_size)}")
    print(f"  Output size: {format_size(output_size)}")

    return True


def format_size(size_bytes):
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def main():
    parser = argparse.ArgumentParser(
        description="Extract subset from large vector datasets",
        epilog="Example: python extract_subset.py bigann_base.bvecs sift10m_base.bvecs 10000000"
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
