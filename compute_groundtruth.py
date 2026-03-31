#!/usr/bin/env python3
"""
Compute exact ground truth (k-NN) for a dataset using FAISS brute-force search.

Usage:
  python compute_groundtruth.py <base_file> <query_file> <output_file> [--k K]

Example:
  python compute_groundtruth.py deep1m_base.fvecs deep1m_query.fvecs deep1m_groundtruth.ivecs --k 100
"""

import argparse
import sys
import os
import numpy as np
from vecs_io import fvecs_mmap, fvecs_read, bvecs_mmap, bvecs_read, ivecs_write


def compute_groundtruth(base_file, query_file, output_file, k):
    """Compute exact k-NN using FAISS brute-force (IndexFlatL2)."""
    try:
        import faiss
    except ImportError:
        print("Error: FAISS is required. Install with: pip install faiss-cpu", file=sys.stderr)
        sys.exit(1)

    base_ext = base_file.rsplit(".", 1)[-1].lower()
    query_ext = query_file.rsplit(".", 1)[-1].lower()

    # Memory-map base vectors (zero-copy for large files)
    print(f"Memory-mapping base vectors from {base_file}...")
    if base_ext == "bvecs":
        xb_raw = bvecs_mmap(base_file)
        xb = np.ascontiguousarray(xb_raw, dtype="float32")
    else:
        xb_raw = fvecs_mmap(base_file)
        xb = np.ascontiguousarray(xb_raw, dtype="float32")
    nb, d = xb.shape
    print(f"  Base: {nb:,} vectors, {d} dimensions")

    # Fully read query vectors (small file)
    print(f"Loading query vectors from {query_file}...")
    if query_ext == "bvecs":
        xq = bvecs_read(query_file).astype("float32")
    else:
        xq = fvecs_read(query_file)
    nq, dq = xq.shape
    print(f"  Query: {nq:,} vectors, {dq} dimensions")

    assert d == dq, f"Dimension mismatch: base={d}, query={dq}"

    print(f"Building FAISS IndexFlatL2 and searching (k={k})...")
    index = faiss.IndexFlatL2(d)
    index.add(xb)
    _, I = index.search(xq, k)

    print(f"Writing ground truth to {output_file}...")
    ivecs_write(output_file, I.astype("int32"))

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
