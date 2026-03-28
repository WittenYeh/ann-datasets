#!/usr/bin/env python3
"""
Convert HDF5 datasets (ann-benchmarks format) to fvecs/ivecs format.

HDF5 datasets from ann-benchmarks.com typically contain:
  - 'train': base vectors (float32)
  - 'test': query vectors (float32)
  - 'neighbors': ground truth neighbor indices (int32)
  - 'distances': ground truth distances (float32)

Usage:
  python hdf5_to_fvecs.py <input.hdf5> <output_prefix>
  python hdf5_to_fvecs.py glove-100-angular.hdf5 glove100

Output files:
  <prefix>_base.fvecs, <prefix>_query.fvecs, <prefix>_groundtruth.ivecs
"""

import argparse
import struct
import sys
import os

try:
    import h5py
except ImportError:
    print("Error: h5py is required. Install with: pip install h5py", file=sys.stderr)
    sys.exit(1)

import numpy as np


def write_fvecs(filename, data):
    """Write float32 vectors in fvecs format."""
    data = np.asarray(data, dtype=np.float32)
    n, d = data.shape
    with open(filename, 'wb') as f:
        for i in range(n):
            f.write(struct.pack('i', d))
            f.write(data[i].tobytes())
            if (i + 1) % 100000 == 0:
                print(f"  Written {i + 1:,} / {n:,} vectors")
    print(f"  -> {filename}: {n:,} vectors, dim={d}")


def write_ivecs(filename, data):
    """Write int32 vectors in ivecs format."""
    data = np.asarray(data, dtype=np.int32)
    n, d = data.shape
    with open(filename, 'wb') as f:
        for i in range(n):
            f.write(struct.pack('i', d))
            f.write(data[i].tobytes())
    print(f"  -> {filename}: {n:,} vectors, dim={d}")


def convert_hdf5(input_file, output_prefix, num_gt_neighbors=100):
    """Convert HDF5 file to fvecs/ivecs files."""
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.", file=sys.stderr)
        return False

    print(f"Opening {input_file}...")
    with h5py.File(input_file, 'r') as f:
        print(f"  Datasets: {list(f.keys())}")

        # Base vectors
        if 'train' in f:
            print(f"\nConverting base vectors ('train')...")
            base = np.array(f['train'], dtype=np.float32)
            write_fvecs(f"{output_prefix}_base.fvecs", base)
        else:
            print("Warning: 'train' dataset not found in HDF5 file.")

        # Query vectors
        if 'test' in f:
            print(f"\nConverting query vectors ('test')...")
            query = np.array(f['test'], dtype=np.float32)
            write_fvecs(f"{output_prefix}_query.fvecs", query)
        else:
            print("Warning: 'test' dataset not found in HDF5 file.")

        # Ground truth neighbors
        if 'neighbors' in f:
            print(f"\nConverting ground truth ('neighbors')...")
            gt = np.array(f['neighbors'], dtype=np.int32)
            # Truncate to num_gt_neighbors if needed
            if gt.shape[1] > num_gt_neighbors:
                gt = gt[:, :num_gt_neighbors]
            write_ivecs(f"{output_prefix}_groundtruth.ivecs", gt)
        else:
            print("Warning: 'neighbors' dataset not found in HDF5 file.")

    print(f"\nConversion complete!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert HDF5 (ann-benchmarks format) to fvecs/ivecs",
        epilog="Example: python hdf5_to_fvecs.py glove-100-angular.hdf5 glove100"
    )
    parser.add_argument("input_file", help="Input HDF5 file")
    parser.add_argument("output_prefix", help="Output file prefix")
    parser.add_argument("-k", "--num-neighbors", type=int, default=100,
                        help="Number of ground truth neighbors to keep (default: 100)")

    args = parser.parse_args()
    success = convert_hdf5(args.input_file, args.output_prefix, args.num_neighbors)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
