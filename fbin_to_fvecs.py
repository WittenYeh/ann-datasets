#!/usr/bin/env python3
"""
Convert .fbin format (BigANN) to .fvecs format (Texmex).

fbin: [uint32 nrows][uint32 ncols][float32 * nrows * ncols]
fvecs: for each row: [int32 dim][float32 * dim]

Uses memory-mapped I/O + chunked writing to avoid OOM on billion-scale files.

Usage:
  python fbin_to_fvecs.py input.fbin output.fvecs [--limit N]
"""

import argparse
import os
import numpy as np

CHUNK_SIZE = 1_000_000  # vectors per chunk


def fbin_to_fvecs(input_file, output_file, limit=None):
    # Read header (8 bytes)
    header = np.fromfile(input_file, dtype="uint32", count=2)
    nrows, ncols = int(header[0]), int(header[1])
    print(f"Input: {input_file}: {nrows:,} vectors x {ncols} dims")

    # Memory-map the data region (skip 8-byte header)
    data = np.memmap(input_file, dtype="float32", mode="r",
                     offset=8, shape=(nrows, ncols))

    if limit is not None and limit < nrows:
        print(f"Limit set: emitting first {limit:,} of {nrows:,} vectors")
        nrows = limit

    print(f"Writing {output_file} in chunks of {CHUNK_SIZE:,}...")
    with open(output_file, "wb") as fout:
        for start in range(0, nrows, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, nrows)
            chunk = np.ascontiguousarray(data[start:end])
            n, d = chunk.shape
            # Build fvecs block: prepend int32 dim to each row
            block = np.empty((n, d + 1), dtype="float32")
            block[:, 0] = np.array([d], dtype="int32").view("float32")
            block[:, 1:] = chunk
            block.tofile(fout)
            print(f"  {end:,} / {nrows:,}")

    print(f"Done. Output: {output_file} ({os.path.getsize(output_file) / 1e9:.2f} GB)")


def main():
    p = argparse.ArgumentParser(description="Convert .fbin → .fvecs")
    p.add_argument("input_file",  help="Input .fbin file")
    p.add_argument("output_file", help="Output .fvecs file")
    p.add_argument("--limit", type=int, default=None,
                   help="Only emit the first N vectors (default: all)")
    args = p.parse_args()
    fbin_to_fvecs(args.input_file, args.output_file, args.limit)


if __name__ == "__main__":
    main()