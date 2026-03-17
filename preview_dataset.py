#!/usr/bin/env python3
"""
Vector dataset preview tool for fvecs, bvecs, and ivecs formats.

Supports:
- fvecs: float32 vectors (4 bytes per element)
- bvecs: uint8 vectors (1 byte per element)
- ivecs: int32 vectors (4 bytes per element)

Usage:
  python preview_dataset.py <filename>
  python preview_dataset.py <filename> --preview 5
"""

import argparse
import struct
import sys
import os


def get_format_info(filename):
    """Determine vector format from file extension."""
    ext = filename.lower().split('.')[-1]
    if ext == 'fvecs':
        return 'fvecs', 'f', 4, 'float32'
    elif ext == 'bvecs':
        return 'bvecs', 'B', 1, 'uint8'
    elif ext == 'ivecs':
        return 'ivecs', 'i', 4, 'int32'
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Expected fvecs, bvecs, or ivecs.")


def read_vector_header(f, elem_size):
    """Read the dimension from vector file header."""
    dim_bytes = f.read(4)
    if len(dim_bytes) < 4:
        return None
    dim = struct.unpack('i', dim_bytes)[0]
    return dim


def inspect_vectors(filename, preview_count=0):
    """Inspect vector file and print statistics."""

    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.", file=sys.stderr)
        return

    try:
        format_name, format_char, elem_size, dtype_name = get_format_info(filename)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return

    file_size = os.path.getsize(filename)

    print(f"\n{'='*60}")
    print(f"Vector File: {os.path.basename(filename)}")
    print(f"{'='*60}")

    with open(filename, 'rb') as f:
        # Read first vector to get dimension
        dim = read_vector_header(f, elem_size)
        if dim is None:
            print("Error: File is empty or corrupted.", file=sys.stderr)
            return

        # Calculate vector size in bytes: 4 bytes (dim) + dim * elem_size
        vec_size_bytes = 4 + dim * elem_size

        # Calculate number of vectors
        num_vecs = file_size // vec_size_bytes

        # Verify file size is consistent
        expected_size = num_vecs * vec_size_bytes
        if expected_size != file_size:
            print(f"Warning: File size mismatch. Expected {expected_size} bytes, got {file_size} bytes.")
            print(f"         File may be corrupted or incomplete.")

        print(f"Format:      {format_name} ({dtype_name})")
        print(f"Dimension:   {dim}")
        print(f"Num Vectors: {num_vecs:,}")
        print(f"File Size:   {format_size(file_size)}")
        print(f"Vector Size: {vec_size_bytes} bytes ({4} + {dim} × {elem_size})")

        # Preview vectors if requested
        if preview_count > 0:
            print(f"\n--- Preview (first {preview_count} vectors) ---")
            f.seek(0)

            for i in range(min(preview_count, num_vecs)):
                # Read dimension
                dim_bytes = f.read(4)
                if len(dim_bytes) < 4:
                    break
                vec_dim = struct.unpack('i', dim_bytes)[0]

                # Read vector data
                vec_data = f.read(vec_dim * elem_size)
                if len(vec_data) < vec_dim * elem_size:
                    break

                # Unpack vector
                vec = struct.unpack(f'{vec_dim}{format_char}', vec_data)

                # Print first few elements
                preview_elems = min(10, len(vec))
                vec_str = ', '.join(f'{v:.4f}' if format_char == 'f' else str(v) for v in vec[:preview_elems])
                if len(vec) > preview_elems:
                    vec_str += ', ...'

                print(f"Vector {i}: [{vec_str}]")

    print(f"{'='*60}\n")


def format_size(size_bytes):
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def main():
    parser = argparse.ArgumentParser(
        description="Preview vector dataset files (fvecs, bvecs, ivecs)",
        epilog="Example: python preview_dataset.py sift_base.fvecs --preview 5"
    )

    parser.add_argument("filename", help="Path to vector file")
    parser.add_argument("-p", "--preview", type=int, default=0, metavar="N",
                        help="Preview first N vectors (default: 0, no preview)")

    args = parser.parse_args()

    inspect_vectors(args.filename, args.preview)


if __name__ == "__main__":
    main()
