#!/usr/bin/env python3
"""Extract random query vectors from a base fvecs/bvecs file.

Randomly samples `num_queries` vectors (without replacement) as query vectors,
removes them from the base set, and writes the compacted base and query files.
Optionally computes ground truth.

Usage:
    python extract_query.py input_base.fvecs base_out.fvecs query_out.fvecs [--num-queries 1000] [--seed 42]
    python extract_query.py input_base.fvecs base_out.fvecs query_out.fvecs --gt gt_out.ivecs --k 100
"""

import argparse
import numpy as np
import sys
from pathlib import Path

# Reuse the shared I/O utilities
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vecs_io import fvecs_mmap, fvecs_write, bvecs_mmap, bvecs_write

# Default: 2M vectors per chunk (~1 GB at dim=128, float32)
DEFAULT_CHUNK = 2_000_000


def _write_base_chunked(data, mask, output_path, ext, chunk_size):
    """Write base vectors in chunks to avoid loading all into memory at once."""
    n, d = data.shape
    written = 0
    with open(output_path, "wb") as f:
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_mask = mask[start:end]
            if not chunk_mask.any():
                continue
            chunk = np.ascontiguousarray(data[start:end][chunk_mask])
            nc = chunk.shape[0]
            if ext == ".fvecs":
                buf = np.empty((nc, d + 1), dtype="int32")
                buf[:, 0] = d
                buf[:, 1:] = chunk.astype("float32").view("int32")
            else:  # .bvecs
                buf = np.empty((nc, d + 4), dtype="uint8")
                buf[:, :4] = np.array([d], dtype="int32").view("uint8")
                buf[:, 4:] = chunk.astype("uint8")
            f.write(buf.tobytes())
            written += nc
            print(f"  Written {written:,} base vectors...")
    return written


def main():
    parser = argparse.ArgumentParser(
        description="Extract random query vectors from a base vector file"
    )
    parser.add_argument("input_base", help="Input base vector file (fvecs or bvecs)")
    parser.add_argument("output_base", help="Output base file (query vectors removed)")
    parser.add_argument("output_query", help="Output query file")
    parser.add_argument(
        "--num-queries", type=int, default=1000,
        help="Number of query vectors to extract (default: 1000)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--gt", default=None, help="Output ground truth file (optional)")
    parser.add_argument("--k", type=int, default=100, help="Ground truth k (default: 100)")
    args = parser.parse_args()

    ext = Path(args.input_base).suffix.lower()
    if ext == ".fvecs":
        data = fvecs_mmap(args.input_base)
        write_fn = fvecs_write
    elif ext == ".bvecs":
        data = bvecs_mmap(args.input_base)
        write_fn = bvecs_write
    else:
        print(f"Error: unsupported format '{ext}', expected .fvecs or .bvecs")
        sys.exit(1)

    n, d = data.shape
    nq = args.num_queries

    if nq >= n:
        print(f"Error: num_queries ({nq}) >= total vectors ({n})")
        sys.exit(1)

    print(f"Input: {n} vectors, {d} dimensions")
    print(f"Extracting {nq} random query vectors (seed={args.seed})...")

    # Generate random indices without replacement
    rng = np.random.default_rng(args.seed)
    query_indices = np.sort(rng.choice(n, size=nq, replace=False))

    # Build mask for base vectors
    mask = np.ones(n, dtype=bool)
    mask[query_indices] = False

    # Write query vectors (always small, safe to load into memory)
    query_vecs = np.array(data[query_indices])
    write_fn(args.output_query, query_vecs)
    print(f"Query: {nq} vectors -> {args.output_query}")

    # Write compacted base vectors in chunks (query vectors removed)
    print(f"Writing base vectors (chunked, chunk_size={DEFAULT_CHUNK:,})...")
    nb_out = _write_base_chunked(data, mask, args.output_base, ext, DEFAULT_CHUNK)
    print(f"Base:  {nb_out:,} vectors -> {args.output_base}")

    # Optionally compute ground truth (delegates to compute_groundtruth.py)
    if args.gt:
        from compute_groundtruth import compute_groundtruth
        compute_groundtruth(args.output_base, args.output_query, args.gt,
                            args.k, DEFAULT_CHUNK)

    print("Done.")


if __name__ == "__main__":
    main()
