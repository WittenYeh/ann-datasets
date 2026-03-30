#!/usr/bin/env python3
"""
Convert BigANN ground truth .bin format to .ivecs format.

bin:   [uint32 nrows][uint32 ncols][int32 * nrows * ncols]
ivecs: for each row: [int32 dim][int32 * dim]

Usage:
  python bin_to_ivecs.py input.bin output.ivecs
"""

import struct
import sys
import os
import numpy as np


def bin_to_ivecs(input_file, output_file):
    with open(input_file, 'rb') as fin:
        nrows = struct.unpack('I', fin.read(4))[0]
        ncols = struct.unpack('I', fin.read(4))[0]
        print(f"Converting {input_file}: {nrows:,} queries x {ncols} neighbors")

        with open(output_file, 'wb') as fout:
            for i in range(nrows):
                row = np.frombuffer(fin.read(ncols * 4), dtype=np.int32)
                fout.write(struct.pack('i', ncols))
                fout.write(row.tobytes())

    print(f"Done. Output: {output_file} ({os.path.getsize(output_file) / 1e6:.1f} MB)")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.bin> <output.ivecs>")
        sys.exit(1)
    bin_to_ivecs(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
