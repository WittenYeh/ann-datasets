#!/usr/bin/env python3
"""
Convert .fbin format (BigANN) to .fvecs format (Texmex).

fbin: [uint32 nrows][uint32 ncols][float32 * nrows * ncols]
fvecs: for each row: [int32 dim][float32 * dim]

Usage:
  python fbin_to_fvecs.py input.fbin output.fvecs
"""

import struct
import sys
import os
import numpy as np


def fbin_to_fvecs(input_file, output_file):
    with open(input_file, 'rb') as fin:
        nrows = struct.unpack('I', fin.read(4))[0]
        ncols = struct.unpack('I', fin.read(4))[0]
        print(f"Converting {input_file}: {nrows:,} vectors x {ncols} dims")

        with open(output_file, 'wb') as fout:
            chunk = 10000
            written = 0
            while written < nrows:
                batch = min(chunk, nrows - written)
                data = np.frombuffer(fin.read(batch * ncols * 4), dtype=np.float32)
                data = data.reshape(batch, ncols)
                for row in data:
                    fout.write(struct.pack('i', ncols))
                    fout.write(row.tobytes())
                written += batch
                if written % 1000000 == 0:
                    print(f"  Progress: {written:,} / {nrows:,}")

    print(f"Done. Output: {output_file} ({os.path.getsize(output_file) / 1e9:.2f} GB)")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.fbin> <output.fvecs>")
        sys.exit(1)
    fbin_to_fvecs(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
