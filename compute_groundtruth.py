#!/usr/bin/env python3
"""
Compute exact ground truth (k-NN) for a dataset using FAISS brute-force search.

Usage:
  python compute_groundtruth.py <base_file> <query_file> <output_file> [--k K]

Example:
  python compute_groundtruth.py deep1m_base.fvecs deep1m_query.fvecs deep1m_groundtruth.ivecs --k 100
"""

import argparse
import struct
import sys
import os
import numpy as np


def read_vecs(filename):
    """Read fvecs or bvecs file into a numpy array."""
    ext = filename.lower().split('.')[-1]

    if ext == 'fvecs':
        with open(filename, 'rb') as f:
            d = struct.unpack('i', f.read(4))[0]
            f.seek(0, 2)
            n = f.tell() // (4 + d * 4)
            f.seek(0)
            vecs = np.zeros((n, d), dtype=np.float32)
            for i in range(n):
                f.read(4)  # skip dim
                vecs[i] = np.frombuffer(f.read(d * 4), dtype=np.float32)
        return vecs

    elif ext == 'bvecs':
        with open(filename, 'rb') as f:
            d = struct.unpack('i', f.read(4))[0]
            f.seek(0, 2)
            n = f.tell() // (4 + d)
            f.seek(0)
            vecs = np.zeros((n, d), dtype=np.float32)
            for i in range(n):
                f.read(4)  # skip dim
                vecs[i] = np.frombuffer(f.read(d), dtype=np.uint8).astype(np.float32)
        return vecs

    else:
        raise ValueError(f"Unsupported format: {ext}")


def write_ivecs(filename, ids):
    """Write integer vectors in ivecs format."""
    n, k = ids.shape
    ids = ids.astype(np.int32)
    with open(filename, 'wb') as f:
        for i in range(n):
            f.write(struct.pack('i', k))
            f.write(ids[i].tobytes())


def compute_groundtruth(base_file, query_file, output_file, k):
    """Compute exact k-NN using FAISS brute-force (IndexFlatL2)."""
    try:
        import faiss
    except ImportError:
        print("Error: FAISS is required. Install with: pip install faiss-cpu", file=sys.stderr)
        sys.exit(1)

    print(f"Loading base vectors from {base_file}...")
    xb = read_vecs(base_file)
    nb, d = xb.shape
    print(f"  Base: {nb:,} vectors, {d} dimensions")

    print(f"Loading query vectors from {query_file}...")
    xq = read_vecs(query_file)
    nq, dq = xq.shape
    print(f"  Query: {nq:,} vectors, {dq} dimensions")

    assert d == dq, f"Dimension mismatch: base={d}, query={dq}"

    print(f"Building FAISS IndexFlatL2 and searching (k={k})...")
    index = faiss.IndexFlatL2(d)
    index.add(xb)

    D, I = index.search(xq, k)

    print(f"Writing ground truth to {output_file}...")
    write_ivecs(output_file, I)

    output_size = os.path.getsize(output_file)
    print(f"Done. {nq:,} queries x {k} neighbors = {output_size / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Compute exact ground truth (k-NN) using FAISS brute-force search"
    )
    parser.add_argument("base_file", help="Base vectors file (fvecs/bvecs)")
    parser.add_argument("query_file", help="Query vectors file (fvecs/bvecs)")
    parser.add_argument("output_file", help="Output ground truth file (ivecs)")
    parser.add_argument("--k", type=int, default=100,
                        help="Number of nearest neighbors (default: 100)")

    args = parser.parse_args()
    compute_groundtruth(args.base_file, args.query_file, args.output_file, args.k)


if __name__ == "__main__":
    main()
