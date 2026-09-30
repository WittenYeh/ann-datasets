#!/usr/bin/env python3
"""Profile min/max Euclidean pair distances and Delta = max/min on CUDA.

Requires NumPy and a CUDA-enabled PyTorch installation. No sampling or ANN is
used: enumerate the upper triangle in square tiles. The default is float64.
GPU memory is O(N * D + block_size**2), rather than O(N**2).

The norm/dot-product distance expansion can lose precision near coincident
points, even in float64. A conservative roundoff bound retains all possible
winning pairs for direct sum((x - y)**2) recomputation. The reported winning
pair is also checked on the CPU with math.fsum. This is exhaustive search with
floating-point distances, not arbitrary-precision real arithmetic. Distinct
rows with equal coordinates correctly have distance zero; only self-pairs are
excluded. Any one extremal pair is returned when there are ties. There is no
early exit on a zero minimum: the maximum still needs the complete scan.

--matmul tf32 enables faster tensor-core screening, with a conservative error
bound for operand rounding AND accumulation. All potentially winning pairs
are recomputed from the ORIGINAL coordinates in float64. --matmul float32 uses
the same approach without TF32. Neither option samples or omits pairs. Use
--compile to fuse distance expansion and min/max reduction with torch.compile
(PyTorch >= 2, Triton), avoiding full-size intermediate distance matrices.

Delta is null in JSON if min=0; delta_status distinguishes an infinite ratio
(max>0) from undefined 0/0 (all coincident). Coordinates are not normalized.

Examples (from the artea-benchmark root):
  python3 third-party/ann-datasets/profile_pair_distance.py \
      --dataset sift-1m --device cuda:0 --output /tmp/sift-distances.json
  python3 third-party/ann-datasets/profile_pair_distance.py \
      --base third-party/ann-datasets/glove-100d/glove100d_base.fvecs
  python3 third-party/ann-datasets/profile_pair_distance.py \
      --dataset deep-10m --matmul tf32 --compile --device cuda:0
"""

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch


def resolve_base(args):
    """Resolve an explicit input or the base entry in datasets.json."""
    def expand(value):
        return Path(os.path.expandvars(os.path.expanduser(str(value))))

    if args.base is not None:
        return expand(args.base).resolve()
    repo = Path(__file__).resolve().parents[2]
    config_path = expand(args.config)
    if not config_path.is_absolute():
        config_path = repo / config_path
    with config_path.open() as stream:
        config = json.load(stream)
    if args.dataset not in config["datasets"]:
        raise ValueError(f"Unknown dataset {args.dataset!r} in {config_path}")
    root = expand(config["root_dir"])
    if not root.is_absolute():
        root = repo / root
    entry = config["datasets"][args.dataset]
    return (root / entry["dataset_dir"] / entry["base_path"]).resolve()


def load_fvecs(path):
    """Validate every header/value and return a contiguous float32 array."""
    path = Path(path)
    if path.suffix.lower() != ".fvecs":
        raise ValueError("Input must be an .fvecs file")
    size = path.stat().st_size
    if size < 4 or size % 4:
        raise ValueError("Empty or truncated fvecs file")
    words = np.memmap(path, dtype="<i4", mode="r")
    dim = int(words[0])
    if dim <= 0 or len(words) % (dim + 1):
        raise ValueError("Invalid dimension or truncated fvecs record")
    records = words.reshape(-1, dim + 1)
    if len(records) < 2:
        raise ValueError("At least two vectors are required")
    for start in range(0, len(records), 65536):
        chunk = records[start:start + 65536]
        if not np.all(chunk[:, 0] == dim):
            raise ValueError("Inconsistent fvecs dimension headers")
        if not np.isfinite(chunk[:, 1:].view("<f4")).all():
            raise ValueError("Coordinates must be finite (no NaN or infinity)")
    return np.array(records[:, 1:].view("<f4"), dtype=np.float32, order="C")


@contextmanager
def matmul_precision(mode):
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = mode == "tf32"
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def tile_extrema(products, a_norms, b_norms, diagonal):
    """Fusible expansion/reduction: FP64 norms and distance arithmetic."""
    distances = products.to(torch.float64) * -2.0 + a_norms[:, None] + b_norms[None, :]
    if diagonal:
        rows = torch.arange(products.shape[0], device=products.device)
        cols = torch.arange(products.shape[1], device=products.device)
        valid = rows[:, None] < cols[None, :]
        minima = torch.where(valid, distances, math.inf).amin(dim=1)
        maxima = torch.where(valid, distances, -math.inf).amax(dim=1)
    else:
        minima, maxima = distances.amin(dim=1), distances.amax(dim=1)
    # Explicit two-stage reduction gives the compiler many independent rows;
    # a single dynamic reduction over an entire 8192^2 tile can serialize work.
    return torch.stack((minima.amin(), maxima.amax()))


def tile_candidate_mask(products, a_norms, b_norms, limits, diagonal):
    """Fusible candidate selection; output is one boolean per pair."""
    distances = products.to(torch.float64) * -2.0 + a_norms[:, None] + b_norms[None, :]
    keep = (distances <= limits[0]) | (distances >= limits[1])
    if diagonal:
        rows = torch.arange(products.shape[0], device=products.device)
        cols = torch.arange(products.shape[1], device=products.device)
        keep &= rows[:, None] < cols[None, :]
    return keep


def candidate_batches(products, a_norms, b_norms, low, high, diagonal,
                      mask=None, count=0):
    """Yield every contender, bounding index storage even for dense ties."""
    if mask is not None:
        flat = mask.view(-1)
        step = flat.numel() if count <= (1 << 20) else (1 << 20)
        for offset in range(0, flat.numel(), step):
            yield torch.nonzero(flat[offset:offset + step], as_tuple=True)[0] + offset
    else:
        # The eager path avoids full-size expansion/mask intermediates.
        flat = products.view(-1)
        for offset in range(0, flat.numel(), 1 << 20):
            end = min(offset + (1 << 20), flat.numel())
            positions = torch.arange(offset, end, device=products.device)
            left, right = positions // len(b_norms), positions % len(b_norms)
            expanded = flat[offset:end].to(torch.float64) * -2.0 + a_norms[left] + b_norms[right]
            keep = (expanded <= low) | (expanded >= high)
            if diagonal:
                keep &= left < right
            yield torch.nonzero(keep, as_tuple=True)[0] + offset


def distance_error_bound(dim, max_norm, mode):
    """Absolute bound on screened squared distances, including FP32/TF32.

    sum(abs(x_k*y_k)) <= ||x||*||y|| <= max_norm. For TF32 use the
    conservative truncation bound u=2^-10 (also covers round-to-nearest).
    Operand errors contribute 2u+u^2, and FP32 dot accumulation contributes
    gamma_(d+2)*(1+u)^2. Multiplication by -2 doubles the dot error. Norms,
    expansion and direct refinement are FP64; include their error as well.
    A small absolute term covers FP32 subnormal/flush-to-zero products.
    """
    eps64 = np.finfo(np.float64).eps
    bound = 16.0 * (dim + 8) * eps64 * max_norm
    if mode != "float64":
        eps32 = np.finfo(np.float32).eps
        if (dim + 2) * eps32 >= 0.01:
            raise ValueError("Dimension too large for FP32 screening; use --matmul float64")
        gamma = (dim + 2) * eps32 / (1.0 - (dim + 2) * eps32)
        u = 2.0 ** -10 if mode == "tf32" else 0.0
        if max_norm * (1 + u) ** 2 >= np.finfo(np.float32).max / 4:
            raise ValueError("FP32 dot products may overflow; use --matmul float64")
        bound += 2.0 * (2 * u + u * u + gamma * (1 + u) ** 2) * max_norm
        bound += 8.0 * dim * np.finfo(np.float32).tiny
    return bound


@torch.inference_mode()
def profile_pair_distance(points, device="cuda:0", block_size=8192,
                          progress_interval=30.0, matmul="float64",
                          compile_reductions=False):
    """Exhaustive min/max over distinct rows; IDs refer to the original input.

    Optional reduced-precision GEMM only screens candidates: error bounds and
    direct FP64 refinement protect both extrema. No normalization/deduplication.
    """
    if points.ndim != 2 or len(points) < 2 or points.shape[1] < 1:
        raise ValueError("Expected at least two vectors with positive dimension")
    if points.dtype != np.float32:
        raise ValueError("Expected float32 coordinates")
    integer_input = matmul != "float64"
    for start in range(0, len(points), 65536):
        part = points[start:start + 65536]
        if not np.isfinite(part).all():
            raise ValueError("Expected finite coordinates")
        if integer_input:
            integer_input = bool(np.equal(part, np.rint(part)).all()
                                 and np.abs(part).max() <= 1024)
    if block_size < 1 or not math.isfinite(progress_interval) or progress_interval < 0:
        raise ValueError("block_size must be positive; progress_interval nonnegative")
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("A CUDA device and CUDA-enabled PyTorch are required")
    if matmul not in ("float64", "float32", "tf32"):
        raise ValueError("matmul must be float64, float32 or tf32")

    started = time.perf_counter()
    n, dim = points.shape
    dtype = torch.float64 if matmul == "float64" else torch.float32
    # Keep original coordinates in FP32; widen only active tiles for FP64
    # GEMM. This avoids doubling the resident dataset on 5M/10M inputs.
    x = torch.from_numpy(points).to(device=device, dtype=torch.float32)
    norms = torch.empty(n, device=device, dtype=torch.float64)
    # Avoid a second N*D GPU allocation for the norm calculation on 10M sets.
    for start in range(0, n, 65536):
        chunk = x[start:start + 65536].to(torch.float64)
        norms[start:start + len(chunk)] = chunk.square().sum(dim=1)
    del chunk
    eps = torch.finfo(torch.float64).eps
    max_norm = float(norms.max().item())
    # Small integers are represented exactly by TF32. If norms < 2^22,
    # Cauchy-Schwarz bounds every partial dot sum by 2^22, so all products
    # and FP32 accumulations are exact integers as well (24-bit precision).
    integer_gemm_exact = integer_input and max_norm < 2.0 ** 22
    error_bound = 0.0 if integer_gemm_exact else distance_error_bound(dim, max_norm, matmul)
    margin = 4.0 * error_bound
    precise_margin = 4.0 * distance_error_bound(dim, max_norm, "float64")
    reduce_tile = torch.compile(tile_extrema, dynamic=True) if compile_reductions else tile_extrema
    make_mask = torch.compile(tile_candidate_mask, dynamic=True) if compile_reductions else None
    tiles = (n + block_size - 1) // block_size
    total_blocks = tiles * (tiles + 1) // 2
    total_pairs = n * (n - 1) // 2
    min_squared, max_squared = math.inf, -math.inf
    min_pair = max_pair = None
    visited_blocks = visited_pairs = refined_pairs = 0
    fp64_rescanned_blocks = 0
    last_progress = time.perf_counter()

    with matmul_precision(matmul):
        for row in range(0, n, block_size):
            a = x[row:row + block_size].to(dtype)
            # Zero-offset norm tensors avoid torch.compile specializing on
            # the changing storage offsets of dataset views.
            an = norms[row:row + len(a)].clone()
            for col in range(row, n, block_size):
                b = x[col:col + block_size].to(dtype)
                bn = norms[col:col + len(b)].clone()
                products = torch.mm(a, b.T)
                block_min, block_max = reduce_tile(products, an, bn, row == col).tolist()
                visited_pairs += (len(a) * (len(a) - 1) // 2 if row == col else len(a) * len(b))
                visited_blocks += 1
                refine_min = (min_squared != 0 and math.isfinite(block_min)
                              and block_min < min_squared + margin)
                refine_max = math.isfinite(block_max) and block_max > max_squared - margin
                local_margin = margin
                mask, count = None, 0
                if (refine_min or refine_max) and make_mask is not None:
                    low = min(min_squared, block_min) + margin if refine_min else -math.inf
                    high = max(max_squared, block_max) - margin if refine_max else math.inf
                    limits = torch.tensor([low, high], device=device, dtype=torch.float64)
                    mask = make_mask(products, an, bn, limits, row == col)
                    count = int(torch.count_nonzero(mask).item())
                if ((refine_min or refine_max) and matmul != "float64" and not integer_gemm_exact
                        and (mask is None or count > 262144)):
                    # Sparse contenders go directly to FP64 coordinate
                    # subtraction. For dense candidates, tighten the bound
                    # with a FP64 GEMM first. This never caps/drops candidates.
                    mask = None
                    products = torch.mm(a.to(torch.float64), b.to(torch.float64).T)
                    block_min, block_max = reduce_tile(products, an, bn, row == col).tolist()
                    local_margin = precise_margin
                    fp64_rescanned_blocks += 1
                    refine_min = (min_squared != 0 and math.isfinite(block_min)
                                  and block_min < min_squared + local_margin)
                    refine_max = math.isfinite(block_max) and block_max > max_squared - local_margin
                if refine_min or refine_max:
                    low = min(min_squared, block_min) + local_margin if refine_min else -math.inf
                    high = max(max_squared, block_max) - local_margin if refine_max else math.inf
                    if mask is None and make_mask is not None:
                        limits = torch.tensor([low, high], device=device, dtype=torch.float64)
                        mask = make_mask(products, an, bn, limits, row == col)
                        count = int(torch.count_nonzero(mask).item())
                    for ids in candidate_batches(products, an, bn, low, high, row == col, mask, count):
                        # FP64 conversion happens BEFORE subtraction (important).
                        for start in range(0, ids.numel(), 8192):
                            selected = ids[start:start + 8192]
                            left, right = selected // len(b), selected % len(b)
                            diff = a[left].to(torch.float64) - b[right].to(torch.float64)
                            squared = diff.square().sum(dim=1)
                            argmin, argmax = int(squared.argmin().item()), int(squared.argmax().item())
                            lo, hi = float(squared[argmin].item()), float(squared[argmax].item())
                            refined_pairs += selected.numel()
                            if lo < min_squared:
                                min_squared = lo
                                min_pair = (row + int(left[argmin].item()), col + int(right[argmin].item()))
                            if hi > max_squared:
                                max_squared = hi
                                max_pair = (row + int(left[argmax].item()), col + int(right[argmax].item()))
                del products, mask

                now = time.perf_counter()
                if progress_interval and now - last_progress >= progress_interval:
                    elapsed = now - started
                    eta = elapsed * (total_blocks - visited_blocks) / visited_blocks
                    print(f"[{visited_blocks}/{total_blocks} blocks, "
                          f"{100 * visited_pairs / total_pairs:.1f}% pairs] "
                          f"min L2={math.sqrt(min_squared):.12g}, "
                          f"max L2={math.sqrt(max_squared) if max_squared >= 0 else 'pending'}, "
                          f"elapsed={elapsed:.1f}s, ETA={eta:.1f}s", flush=True)
                    last_progress = now

    torch.cuda.synchronize(device)
    if min_pair is None or max_pair is None:
        raise RuntimeError("No finite pair distance found")
    cpu_squared = []
    for pair, squared in ((min_pair, min_squared), (max_pair, max_squared)):
        i, j = pair
        value = math.fsum((float(u) - float(v)) ** 2 for u, v in zip(points[i], points[j]))
        if not math.isclose(value, squared, rel_tol=8 * dim * eps, abs_tol=0.0):
            raise RuntimeError("GPU extremum disagrees with CPU recomputation")
        cpu_squared.append(value)
    minimum, maximum = (math.sqrt(value) for value in cpu_squared)
    return {
        "num_vectors": n,
        "dimension": dim,
        "metric": "euclidean",
        "min_distance": minimum,
        "max_distance": maximum,
        "min_squared_distance": cpu_squared[0],
        "max_squared_distance": cpu_squared[1],
        "delta": maximum / minimum if minimum else None,
        "delta_status": "finite" if minimum else ("infinite" if maximum else "undefined"),
        "min_pair_ids": list(min_pair),
        "max_pair_ids": list(max_pair),
        "pair_id_convention": "zero-based rows of the input base file",
        "method": "tiled exhaustive CUDA scan, bounded screening, direct FP64 refinement",
        "matmul": matmul,
        "distance_dtype": "float64",
        "storage_dtype": "float32",
        "compiled_reductions": compile_reductions,
        "integer_gemm_exact": integer_gemm_exact,
        "total_unique_pairs": total_pairs,
        "evaluated_unique_pairs": visited_pairs,
        "completed_blocks": visited_blocks,
        "total_blocks": total_blocks,
        "directly_refined_pairs": refined_pairs,
        "fp64_rescanned_blocks": fp64_rescanned_blocks,
        "roundoff_error_bound_squared": error_bound,
        "cpu_pair_verification": True,
        "gpu_min_squared_distance": min_squared,
        "gpu_max_squared_distance": max_squared,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "block_size": block_size,
        "compute_seconds": time.perf_counter() - started,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cpu_threads": torch.get_num_threads(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
    }


def save_report(path, report):
    """Publish a complete JSON report atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base", type=Path, help="Input base .fvecs file")
    source.add_argument("--dataset", help="Dataset name in configs/datasets.json")
    parser.add_argument("--config", default="configs/datasets.json",
                        help="Config path; relative paths are resolved from benchmark root")
    parser.add_argument("--device", default="cuda:0", help="CUDA device (default: cuda:0)")
    parser.add_argument("--block-size", type=int, default=8192,
                        help="Square distance tile width; reduce on CUDA OOM (default: 8192)")
    parser.add_argument("--matmul", choices=("float64", "float32", "tf32"), default="float64",
                        help="Screening GEMM precision; all contenders are refined in FP64")
    parser.add_argument("--compile", action="store_true", dest="compile_reductions",
                        help="Fuse expansion/min/max reductions with torch.compile")
    parser.add_argument("--threads", type=int, default=8, help="PyTorch CPU threads (default: 8)")
    parser.add_argument("--progress-interval", type=float, default=30.0,
                        help="Seconds between progress reports; 0 disables (default: 30)")
    parser.add_argument("--output", type=Path, help="Optional JSON report")
    args = parser.parse_args()
    try:
        if args.threads < 1:
            raise ValueError("--threads must be positive")
        torch.set_num_threads(args.threads)
        base = resolve_base(args)
        if args.output and args.output.resolve() == base:
            raise ValueError("Output report must not overwrite the input dataset")
        started = time.perf_counter()
        points = load_fvecs(base)
        load_seconds = time.perf_counter() - started
        print(f"Base: {base}\nVectors: {len(points):,}, dimension: {points.shape[1]}\n"
              f"Device: {args.device}, GEMM: {args.matmul}, FP64 refinement, "
              f"block size: {args.block_size}", flush=True)
        result = profile_pair_distance(points, args.device, args.block_size,
                                       args.progress_interval, args.matmul, args.compile_reductions)
        result.update(base_path=str(base), dataset=args.dataset, load_seconds=load_seconds)
        if args.output:
            save_report(args.output, result)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
