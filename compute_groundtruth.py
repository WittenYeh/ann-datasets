#!/usr/bin/env python3
"""
Compute exact ground truth (k-NN) for a dataset using FAISS brute-force search.

For large base sets the search is performed in chunks so that only one chunk
resides in RAM at a time, keeping peak memory bounded.

Usage:
  python compute_groundtruth.py <base_file> <query_file> <output_file> [--k K] [--chunk-size N]

Example:
  python compute_groundtruth.py deep10m_base.fvecs deep10m_query.fvecs deep10m_groundtruth.ivecs --k 100
"""

import argparse
import sys
import os
import numpy as np
from vecs_io import fvecs_mmap, fvecs_read, bvecs_mmap, bvecs_read, ivecs_write

# Default: 2M vectors per chunk. At dim=128, float32, this is ~1 GB RAM.
DEFAULT_CHUNK = 2_000_000


def _search_chunked(xb_mmap, xq, k, chunk_size, is_bvecs=False):
    """Brute-force k-NN in chunks to avoid loading the entire base into RAM."""
    import faiss

    nq = xq.shape[0]
    nb = xb_mmap.shape[0]
    d = xq.shape[1]

    # Accumulate the global top-k distances and IDs across all chunks.
    # Initialise with +inf so that any real distance wins.
    all_D = np.full((nq, k), np.inf, dtype="float32")
    all_I = np.full((nq, k), -1, dtype="int64")

    for start in range(0, nb, chunk_size):
        end = min(start + chunk_size, nb)
        print(f"  Searching chunk [{start:,} .. {end:,}) / {nb:,}")

        # Materialise *only this chunk* into contiguous float32 RAM
        chunk = np.ascontiguousarray(xb_mmap[start:end], dtype="float32")

        index = faiss.IndexFlatL2(d)
        index.add(chunk)
        D_chunk, I_chunk = index.search(xq, k)

        # Shift local IDs to global IDs
        I_chunk += start

        # Merge: for each query, keep the k smallest distances
        merged_D = np.concatenate([all_D, D_chunk], axis=1)
        merged_I = np.concatenate([all_I, I_chunk], axis=1)
        sel = np.argpartition(merged_D, k, axis=1)[:, :k]
        rows = np.arange(nq)[:, None]
        all_D = merged_D[rows, sel]
        all_I = merged_I[rows, sel]

    # Final sort within the k results
    order = np.argsort(all_D, axis=1)
    rows = np.arange(nq)[:, None]
    return all_I[rows, order]


def compute_groundtruth(base_file, query_file, output_file, k, chunk_size):
    try:
        import faiss
    except ImportError:
        print("Error: FAISS is required. Install with: pip install faiss-cpu", file=sys.stderr)
        sys.exit(1)

    base_ext = base_file.rsplit(".", 1)[-1].lower()
    query_ext = query_file.rsplit(".", 1)[-1].lower()

    # Memory-map base vectors (no RAM until a chunk is materialised)
    print(f"Memory-mapping base vectors from {base_file}...")
    is_bvecs = base_ext == "bvecs"
    xb_mmap = bvecs_mmap(base_file) if is_bvecs else fvecs_mmap(base_file)
    nb, d = xb_mmap.shape
    print(f"  Base: {nb:,} vectors, {d} dimensions")

    # Fully read query vectors (always small)
    print(f"Loading query vectors from {query_file}...")
    if query_ext == "bvecs":
        xq = bvecs_read(query_file).astype("float32")
    else:
        xq = fvecs_read(query_file)
    nq, dq = xq.shape
    print(f"  Query: {nq:,} vectors, {dq} dimensions")
    assert d == dq, f"Dimension mismatch: base={d}, query={dq}"

    # Decide whether to use chunked search
    if nb <= chunk_size:
        print(f"Building FAISS IndexFlatL2 and searching (k={k})...")
        xb = np.ascontiguousarray(xb_mmap, dtype="float32")
        index = faiss.IndexFlatL2(d)
        index.add(xb)
        del xb  # free after add copies internally
        _, I = index.search(xq, k)
    else:
        print(f"Chunked search (k={k}, chunk_size={chunk_size:,})...")
        I = _search_chunked(xb_mmap, xq, k, chunk_size, is_bvecs)

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
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK,
                        help=f"Max base vectors per chunk (default: {DEFAULT_CHUNK:,})")

    args = parser.parse_args()
    compute_groundtruth(args.base_file, args.query_file, args.output_file,
                        args.k, args.chunk_size)


if __name__ == "__main__":
    main()
