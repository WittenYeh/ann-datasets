#!/usr/bin/env python3
"""
Convert .bvecs format (Texmex, uint8) to .fvecs format (Texmex, float32).

bvecs: per row [int32 dim][uint8  * dim]
fvecs: per row [int32 dim][float32 * dim]

Widens uint8 -> float32. The cast is LOSSLESS for the SIFT/BIGANN family:
values 0..255 are exactly representable in float32, and squared-L2 over
128 dims stays < 2^24, so neighbour rankings (and any precomputed ground
truth) are identical to the integer metric. Use this to make the BIGANN
family loadable by readers that only accept .fvecs/.ivecs -- e.g. the
artea VectorArray, which cannot read .bvecs.

Memory-mapped input + chunked write keep peak RAM bounded so the script
streams multi-GB files.

Usage:
  python bvecs_to_fvecs.py input.bvecs output.fvecs [--limit N]
"""

import argparse
import os
from vecs_io import bvecs_mmap, fvecs_write_chunked


def bvecs_to_fvecs(input_file, output_file, limit=None):
    # Read-only (n, d) uint8 view over the Texmex per-row layout.
    data = bvecs_mmap(input_file)
    nrows, ncols = data.shape
    print(f"Input: {input_file}: {nrows:,} vectors x {ncols} dims (uint8)")

    if limit is not None and limit < nrows:
        print(f"Limit set: emitting first {limit:,} of {nrows:,} vectors")
        data = data[:limit]

    print(f"Writing {output_file} in chunks...")
    fvecs_write_chunked(output_file, data)

    print(f"Done. Output: {output_file} ({os.path.getsize(output_file) / 1e9:.2f} GB)")


def main():
    p = argparse.ArgumentParser(description="Convert .bvecs -> .fvecs")
    p.add_argument("input_file",  help="Input .bvecs file")
    p.add_argument("output_file", help="Output .fvecs file")
    p.add_argument("--limit", type=int, default=None,
                   help="Only emit the first N vectors (default: all)")
    args = p.parse_args()
    bvecs_to_fvecs(args.input_file, args.output_file, args.limit)


if __name__ == "__main__":
    main()
