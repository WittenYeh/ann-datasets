#!/usr/bin/env python3
"""
Convert .i8bin format (BigANN) to .fvecs format (Texmex).

i8bin: [uint32 nrows][uint32 ncols][int8 * nrows * ncols]
fvecs: for each row: [int32 dim][float32 * dim]

Same single-header layout as .fbin, but each value is one signed byte.
We widen int8 → float32 row-by-row in chunks; this keeps peak RAM
bounded so the script can stream multi-GB files.

Useful for the Microsoft SpaceV-1B family, whose base set ships as
i8bin (100-dim, signed -128..127). When the on-disk file is a partial
byte-range download (e.g. the first 10M of the 1B base), pass --limit
to clamp the row count and the script will only mmap as many rows as
actually exist on disk.

Usage:
  python i8bin_to_fvecs.py input.i8bin output.fvecs [--limit N]
"""

import argparse
import os
import numpy as np

CHUNK_SIZE = 1_000_000  # vectors per chunk


def i8bin_to_fvecs(input_file, output_file, limit=None):
    header = np.fromfile(input_file, dtype="uint32", count=2)
    nrows, ncols = int(header[0]), int(header[1])
    print(f"Input: {input_file}: {nrows:,} vectors x {ncols} dims (int8)")

    if limit is not None and limit < nrows:
        print(f"Limit set: emitting first {limit:,} of {nrows:,} vectors")
        nrows = limit

    # Memory-map the int8 region (offset past the 8-byte header).
    data = np.memmap(input_file, dtype="int8", mode="r",
                     offset=8, shape=(nrows, ncols))

    print(f"Writing {output_file} in chunks of {CHUNK_SIZE:,}...")
    with open(output_file, "wb") as fout:
        for start in range(0, nrows, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, nrows)
            chunk_i8 = np.ascontiguousarray(data[start:end])
            n, d = chunk_i8.shape
            chunk_f32 = chunk_i8.astype("float32")
            # fvecs row layout: [int32 dim][float32 × dim]
            block = np.empty((n, d + 1), dtype="float32")
            block[:, 0] = np.array([d], dtype="int32").view("float32")
            block[:, 1:] = chunk_f32
            block.tofile(fout)
            print(f"  {end:,} / {nrows:,}")

    print(f"Done. Output: {output_file} ({os.path.getsize(output_file) / 1e9:.2f} GB)")


def main():
    p = argparse.ArgumentParser(description="Convert .i8bin → .fvecs")
    p.add_argument("input_file",  help="Input .i8bin file")
    p.add_argument("output_file", help="Output .fvecs file")
    p.add_argument("--limit", type=int, default=None,
                   help="Only emit the first N vectors (default: all)")
    args = p.parse_args()
    i8bin_to_fvecs(args.input_file, args.output_file, args.limit)


if __name__ == "__main__":
    main()
