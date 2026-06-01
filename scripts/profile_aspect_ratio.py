#!/usr/bin/env python3
"""Compute a dataset's EXACT aspect ratio on the GPU (PyTorch).

Aspect ratio = (max pairwise distance) / (min pairwise distance) over ALL base
points, in Euclidean (L2) distance. This is an O(N^2) all-pairs computation, so
it is done in row-tiles: for each tile of rows we compute its distances to every
point on the GPU and fold them into running max / min reductions — only an
O(tile x N) distance block is ever materialised.

Two "min" notions are reported because duplicate / coincident points make the
raw minimum 0 (degenerate aspect ratio = inf):
  - min_pairwise      : smallest distance between two DISTINCT indices (may be 0)
  - min_pairwise_pos  : smallest STRICTLY-POSITIVE distance (closest distinct
                        *locations*) — use this for a meaningful aspect ratio.

Distances are computed via the squared-L2 expansion
``||x||^2 + ||y||^2 - 2 x.y`` (fast, one matmul per tile). In float32 this loses
precision near coincident points; pass --double for an exact closest-pair when
the dataset has near-duplicates.

Datasets are resolved from configs/datasets.json (same convention as
scripts/compute_lid.py), or pass --base directly.

Usage:
    python profile_aspect_ratio.py --dataset sift-1m
    python profile_aspect_ratio.py --dataset glove-100d --block 2048 --double
    python profile_aspect_ratio.py --base /path/to/base.fvecs --max-points 1000000
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:                       # graceful no-op if tqdm isn't installed
    def tqdm(it, **kwargs):
        return it

# vecs_io lives in the ann-datasets root (the parent of this scripts/ dir).
ANN_DATASET_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ANN_DATASET_DIR))
from vecs_io import read_by_ext, mmap_by_ext  # noqa: E402


def _resolve_base_path(args) -> Path:
    """Return the base-vectors file path, from --base or configs/datasets.json."""
    if args.base:
        return Path(os.path.expanduser(os.path.expandvars(args.base)))
    if not args.dataset:
        sys.exit("ERROR: pass either --base <file> or --dataset <name>.")
    repo_root = ANN_DATASET_DIR.parent.parent  # .../artea-benchmark
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (repo_root / config_path).resolve()
    with open(config_path) as f:
        cfg = json.load(f)
    if args.dataset not in cfg["datasets"]:
        sys.exit(f"ERROR: dataset '{args.dataset}' not in {config_path}. "
                 f"Known: {sorted(cfg['datasets'])}")
    # root_dir may use $HOME / ~ ; expand it (do NOT join with repo_root when absolute).
    root = Path(os.path.expanduser(os.path.expandvars(cfg["root_dir"])))
    if not root.is_absolute():
        root = (repo_root / root).resolve()
    entry = cfg["datasets"][args.dataset]
    return root / entry["dataset_dir"] / entry["base_path"]


def _load_points(base_path: Path, max_points: int, seed: int) -> np.ndarray:
    """Load base vectors as a contiguous float32 (n, d) array, optionally
    subsampling to at most max_points rows (uniformly at random)."""
    xb = mmap_by_ext(str(base_path))          # zero-copy (n, d) view
    n = xb.shape[0]
    if 0 < max_points < n:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=max_points, replace=False))
        print(f"  subsampling {max_points:,} / {n:,} points (seed={seed}) "
              f"— result is the aspect ratio of the SAMPLE, not the full set")
        return np.ascontiguousarray(xb[idx], dtype=np.float32)
    return np.ascontiguousarray(xb, dtype=np.float32)


def compute_aspect_ratio(X, block: int, pos_eps: float):
    """Tiled all-pairs max / min over X (a torch tensor on the target device).

    Returns (max_dist, min_dist, min_dist_pos, n_zero_pairs) where distances are
    Euclidean (sqrt of the squared-L2 reduction). Self-pairs (i == i) are
    excluded; min_dist is over distinct indices (can be 0 for duplicates),
    min_dist_pos is the smallest distance strictly greater than pos_eps.
    """
    import torch

    n = X.shape[0]
    sq_norms = (X * X).sum(dim=1)             # (n,)
    INF = torch.tensor(float("inf"), device=X.device, dtype=X.dtype)

    max_d2 = torch.zeros((), device=X.device, dtype=X.dtype)
    min_d2 = INF.clone()
    min_d2_pos = INF.clone()
    n_zero_pairs = 0

    pos_eps2 = float(pos_eps) ** 2

    starts = range(0, n, block)
    n_tiles = (n + block - 1) // block
    for s in tqdm(starts, total=n_tiles, desc="all-pairs", unit="tile"):
        e = min(s + block, n)
        a = X[s:e]                            # (b, d)
        # squared L2 block: ||a||^2 + ||y||^2 - 2 a.y   → (b, n)
        d2 = sq_norms[s:e].unsqueeze(1) + sq_norms.unsqueeze(0) - 2.0 * (a @ X.t())
        d2.clamp_(min=0.0)                    # kill tiny negatives from cancellation

        # Exclude the diagonal (self-distance) from BOTH reductions.
        rows = torch.arange(e - s, device=X.device)
        d2[rows, s + rows] = INF              # +inf → ignored by min; for max we
        block_max = d2.masked_fill(torch.isinf(d2), 0.0).max()  # treat inf as 0
        max_d2 = torch.maximum(max_d2, block_max)

        block_min = d2.min()                  # smallest distinct-pair distance
        min_d2 = torch.minimum(min_d2, block_min)

        # zero (coincident) pairs and strictly-positive minimum
        zero_mask = d2 <= pos_eps2
        n_zero_pairs += int(zero_mask.sum().item())
        d2_pos = d2.masked_fill(zero_mask, float("inf"))
        block_min_pos = d2_pos.min()
        min_d2_pos = torch.minimum(min_d2_pos, block_min_pos)

    sqrt = lambda t: float(t.clamp(min=0.0).sqrt().item())
    # n_zero_pairs double-counts (i,j) and (j,i); report symmetric-pair count.
    return (sqrt(max_d2), sqrt(min_d2), sqrt(min_d2_pos), n_zero_pairs // 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_argument_group("dataset source (pick one)")
    src.add_argument("--dataset", default=None,
                     help="Dataset name in configs/datasets.json (uses its base file).")
    src.add_argument("--base", default=None,
                     help="Direct path to a base .fvecs/.bvecs/.ivecs file.")
    parser.add_argument("--config", default="configs/datasets.json",
                        help="datasets.json path (default: configs/datasets.json).")
    parser.add_argument("--block", type=int, default=16384,
                        help="Row-tile size; GPU holds one (block x N) fp32 "
                             "distance matrix (= block*N*4 bytes, x2 for "
                             "--double). Raise for speed if VRAM allows; lower "
                             "it on OOM. Default: 16384.")
    parser.add_argument("--max-points", type=int, default=0,
                        help="Cap #points by random subsampling (0 = use all). "
                             "Note: O(N^2) — for 10M+ points subsample or expect "
                             "a long run.")
    parser.add_argument("--double", action="store_true",
                        help="Use float64 (exact closest-pair near duplicates; "
                             "~2x memory, slower). Default: float32.")
    parser.add_argument("--pos-eps", type=float, default=0.0,
                        help="Distances <= pos-eps count as 'coincident' (0) for "
                             "min_pairwise_pos. Default: 0 (strictly positive).")
    parser.add_argument("--device", default="cuda",
                        help="Torch device (default: cuda).")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="If CUDA is unavailable, run on CPU instead of "
                             "erroring out (O(N^2) on CPU is very slow — pair "
                             "with --max-points).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Subsampling seed (default: 42).")
    parser.add_argument("--output", default=None,
                        help="Optional JSON file to write the result to.")
    args = parser.parse_args()

    import torch
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        diag = (
            "CUDA was requested but torch.cuda.is_available() is False.\n"
            f"  torch              = {torch.__version__}\n"
            f"  torch.version.cuda = {torch.version.cuda}\n"
            f"  cuda.device_count  = {torch.cuda.device_count()}\n"
        )
        if torch.version.cuda is None:
            diag += ("  → This is a CPU-only torch build. Install a CUDA build, e.g.\n"
                     "      pip install torch --index-url https://download.pytorch.org/whl/cu124\n")
        else:
            diag += ("  → torch is a CUDA build, but no GPU/driver is visible on this host\n"
                     "    (e.g. `nvidia-smi` missing, CPU-only / login node, or\n"
                     "    CUDA_VISIBLE_DEVICES hides all GPUs). Run on a GPU node.\n")
        if not args.allow_cpu:
            sys.exit("ERROR: " + diag +
                     "Pass --allow-cpu to run on CPU anyway (slow; pair with --max-points).")
        print("WARNING: " + diag + "Running on CPU (--allow-cpu).")
        device = "cpu"
    dtype = torch.float64 if args.double else torch.float32

    base_path = _resolve_base_path(args)
    if not base_path.exists():
        sys.exit(f"ERROR: base file not found: {base_path}")
    name = args.dataset or base_path.name
    print(f"Dataset: {name}\n  base file: {base_path}")

    pts = _load_points(base_path, args.max_points, args.seed)
    n, d = pts.shape
    print(f"  points: {n:,}  dim: {d}  device: {device}  dtype: {dtype}")
    if n > 200_000 and args.max_points == 0:
        print(f"  NOTE: O(N^2) = ~{n * n / 1e12:.1f}e12 pair-distances; this may "
              f"take a while. Use --max-points to subsample.")

    X = torch.from_numpy(pts).to(device=device, dtype=dtype)

    t0 = time.time()
    max_dist, min_dist, min_dist_pos, n_zero = compute_aspect_ratio(
        X, block=args.block, pos_eps=args.pos_eps)
    elapsed = time.time() - t0

    ar_raw = (max_dist / min_dist) if min_dist > 0 else float("inf")
    ar_pos = (max_dist / min_dist_pos) if min_dist_pos > 0 else float("inf")

    print(f"\nAspect ratio for {name} (L2, {n:,} points, {elapsed:.1f}s):")
    print(f"  max pairwise distance      : {max_dist:.6g}")
    print(f"  min pairwise distance      : {min_dist:.6g}"
          + (f"   ({n_zero:,} coincident pair(s))" if n_zero else ""))
    print(f"  min pairwise distance (>0) : {min_dist_pos:.6g}")
    print(f"  aspect ratio  max/min      : {ar_raw:.6g}"
          + ("   (inf: duplicate points)" if n_zero else ""))
    print(f"  aspect ratio  max/min(>0)  : {ar_pos:.6g}   <-- recommended")

    if args.output:
        out = {
            "dataset": name, "base_path": str(base_path),
            "n_points": int(n), "dim": int(d), "metric": "l2",
            "dtype": "float64" if args.double else "float32",
            "subsampled": bool(args.max_points and args.max_points < n),
            "max_dist": max_dist, "min_dist": min_dist,
            "min_dist_pos": min_dist_pos, "n_coincident_pairs": int(n_zero),
            "aspect_ratio_max_over_min": ar_raw,
            "aspect_ratio_max_over_min_pos": ar_pos,
            "elapsed_s": elapsed,
        }
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
