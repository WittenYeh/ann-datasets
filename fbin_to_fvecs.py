#!/usr/bin/env python3
"""
Convert .fbin format (BigANN) to .fvecs format (Texmex).

fbin: [uint32 nrows][uint32 ncols][float32 * nrows * ncols]
fvecs: for each row: [int32 dim][float32 * dim]

Uses memory-mapped I/O + chunked writing to avoid OOM on billion-scale files.

Usage:
  python fbin_to_fvecs.py input.fbin output.fvecs
"""

import sys
import os
import numpy as np

CHUNK_SIZE = 1_000_000  # vectors per chunk


def fbin_to_fvecs(input_file, output_file):
    # Read header (8 bytes)
    header = np.fromfile(input_file, dtype="uint32", count=2)
    nrows, ncols = int(header[0]), int(header[1])
    print(f"Input: {input_file}: {nrows:,} vectors x {ncols} dims")

    # Memory-map the data region (skip 8-byte header)
    data = np.memmap(input_file, dtype="float32", mode="r",
                     offset=8, shape=(nrows, ncols))

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
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.fbin> <output.fvecs>")
        sys.exit(1)
    fbin_to_fvecs(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()