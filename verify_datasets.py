#!/usr/bin/env python3
"""Verify integrity of all datasets in the ann-dataset repository.

Auto-discovers datasets by scanning subdirectories for *_base.fvecs / *_base.bvecs
files, then verifies:
  1. File existence (base, query, groundtruth)
  2. File integrity (correct fvecs/bvecs/ivecs format, no truncation)
  3. Dimension consistency (base dim == query dim)
  4. Ground truth validity (row count == query count, IDs in range)
  5. GT correctness sampling (--verify-gt): brute-force on sampled queries

Usage:
    python verify_datasets.py
    python verify_datasets.py --verify-gt [--gt-samples 20]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vecs_io import fvecs_mmap, ivecs_mmap, bvecs_mmap


def _mmap_vecs(filepath):
    """Memory-map a vecs file, return (data_view, n, d)."""
    ext = filepath.suffix.lower()
    if ext == ".fvecs":
        data = fvecs_mmap(str(filepath))
    elif ext == ".bvecs":
        data = bvecs_mmap(str(filepath))
    elif ext == ".ivecs":
        data = ivecs_mmap(str(filepath))
    else:
        raise ValueError(f"Unsupported format: {ext}")
    return data, data.shape[0], data.shape[1]


CHUNK_SIZE = 2_000_000


def discover_datasets(root):
    """Scan subdirectories for datasets. Returns list of dicts."""
    datasets = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith(".") or d.name == "__pycache__":
            continue
        base_files = list(d.glob("*_base.fvecs")) + list(d.glob("*_base.bvecs"))
        if not base_files:
            continue
        base = base_files[0]
        prefix = base.stem.replace("_base", "")
        ext = base.suffix

        query_candidates = [
            d / f"{prefix}_query{ext}",
            d / f"{prefix}_query.fvecs",
        ]
        query = next((q for q in query_candidates if q.exists()), None)

        gt_candidates = [
            d / f"{prefix}_groundtruth.ivecs",
        ]
        gt = next((g for g in gt_candidates if g.exists()), None)

        datasets.append({
            "name": d.name,
            "base": base,
            "query": query,
            "gt": gt,
        })
    return datasets


def verify_gt_samples(base_path, query_path, gt_path, num_samples):
    """Brute-force verify GT correctness on sampled queries."""
    import faiss
    xb = fvecs_mmap(str(base_path))
    xq_all = fvecs_mmap(str(query_path))
    gt_all = ivecs_mmap(str(gt_path))

    nb, d = xb.shape
    nq = xq_all.shape[0]
    k = gt_all.shape[1]
    num_samples = min(num_samples, nq)

    rng = np.random.default_rng(42)
    sample_ids = np.sort(rng.choice(nq, size=num_samples, replace=False))
    xq = np.ascontiguousarray(xq_all[sample_ids], dtype="float32")

    if nb <= CHUNK_SIZE:
        xb_f32 = np.ascontiguousarray(xb, dtype="float32")
        index = faiss.IndexFlatL2(d)
        index.add(xb_f32)
        _, bf_ids = index.search(xq, k)
    else:
        from compute_groundtruth import _search_chunked
        bf_ids = _search_chunked(xb, xq, k, CHUNK_SIZE)

    bf_ids = bf_ids.astype(np.int32)
    passed = sum(
        1 for i in range(num_samples)
        if set(gt_all[sample_ids[i]]) == set(bf_ids[i])
    )
    return passed, num_samples


def verify_one(ds, verify_gt=False, gt_samples=20):
    """Verify a single dataset. Returns result dict."""
    r = {
        "name": ds["name"],
        "base_n": "-", "base_d": "-",
        "query_n": "-", "query_d": "-",
        "gt_n": "-", "gt_k": "-",
        "gt_check": "-",
        "errors": [], "warnings": [],
    }

    # Base
    base_n, base_d = None, None
    if ds["base"] and ds["base"].exists():
        try:
            _, base_n, base_d = _mmap_vecs(ds["base"])
            r["base_n"] = f"{base_n:,}"
            r["base_d"] = str(base_d)
        except Exception as e:
            r["errors"].append(f"base corrupt: {e}")
    else:
        r["errors"].append("base missing")

    # Query
    query_n, query_d = None, None
    if ds["query"] and ds["query"].exists():
        try:
            _, query_n, query_d = _mmap_vecs(ds["query"])
            r["query_n"] = f"{query_n:,}"
            r["query_d"] = str(query_d)
        except Exception as e:
            r["errors"].append(f"query corrupt: {e}")
    else:
        r["warnings"].append("query missing")

    # GT
    gt_n, gt_k = None, None
    if ds["gt"] and ds["gt"].exists():
        try:
            _, gt_n, gt_k = _mmap_vecs(ds["gt"])
            r["gt_n"] = f"{gt_n:,}"
            r["gt_k"] = str(gt_k)
        except Exception as e:
            r["errors"].append(f"gt corrupt: {e}")
    else:
        r["warnings"].append("gt missing")

    # Dimension consistency
    if base_d is not None and query_d is not None and base_d != query_d:
        r["errors"].append(f"dim mismatch: base={base_d}, query={query_d}")

    # GT validity
    if gt_n is not None and query_n is not None and gt_n != query_n:
        r["errors"].append(f"gt rows ({gt_n}) != query count ({query_n})")

    if gt_n is not None and base_n is not None:
        try:
            gt_data = ivecs_mmap(str(ds["gt"]))
            gt_max = int(np.max(gt_data))
            if gt_max >= base_n:
                r["errors"].append(f"gt ID {gt_max} >= base count {base_n}")
            if int(np.min(gt_data)) < 0:
                r["errors"].append("gt has negative ID")
        except Exception:
            pass

    # GT sampling check
    if verify_gt and not r["errors"] and base_n and query_n and gt_n:
        try:
            passed, total = verify_gt_samples(
                ds["base"], ds["query"], ds["gt"], gt_samples)
            r["gt_check"] = f"{passed}/{total}"
            if passed < total:
                r["errors"].append(f"gt check: {passed}/{total}")
        except Exception as e:
            r["gt_check"] = "ERR"
            r["errors"].append(f"gt check: {e}")

    return r


def main():
    parser = argparse.ArgumentParser(description="Verify ann-dataset integrity")
    parser.add_argument("--verify-gt", action="store_true",
                        help="Brute-force verify GT on sampled queries")
    parser.add_argument("--gt-samples", type=int, default=20,
                        help="Queries to sample for GT check (default: 20)")
    args = parser.parse_args()

    if args.verify_gt:
        try:
            import faiss  # noqa: F401
        except ImportError:
            print("Error: --verify-gt requires faiss. Install: pip install faiss-cpu")
            sys.exit(1)

    root = Path(__file__).resolve().parent
    datasets = discover_datasets(root)

    if not datasets:
        print("No datasets found. Run 'make all' in a dataset subdirectory first.")
        sys.exit(1)

    console = Console()
    results = []
    for ds in datasets:
        if args.verify_gt:
            console.print(f"[bold]{ds['name']}[/bold]:")
        results.append(verify_one(ds, args.verify_gt, args.gt_samples))

    # Build table
    table = Table(title="Dataset Verification Report", show_lines=True)
    table.add_column("Dataset", style="bold cyan", no_wrap=True)
    table.add_column("Dim", justify="right")
    table.add_column("# Base", justify="right")
    table.add_column("# Query", justify="right")
    table.add_column("GT k", justify="right")
    if args.verify_gt:
        table.add_column("GT Check", justify="center")
    table.add_column("Status", min_width=12)

    ok = 0
    for r in results:
        if r["errors"]:
            status = "[bold red]FAIL[/bold red]\n" + "\n".join(
                f"[red]  - {e}[/red]" for e in r["errors"]
            )
            if r["warnings"]:
                status += "\n" + "\n".join(
                    f"[yellow]  - {w}[/yellow]" for w in r["warnings"]
                )
        elif r["warnings"]:
            status = "[bold yellow]WARN[/bold yellow]\n" + "\n".join(
                f"[yellow]  - {w}[/yellow]" for w in r["warnings"]
            )
            ok += 1
        else:
            status = "[bold green]OK[/bold green]"
            ok += 1

        row = [r["name"], r["base_d"], r["base_n"], r["query_n"], r["gt_k"]]
        if args.verify_gt:
            gt_check = r["gt_check"]
            if gt_check == "-":
                row.append("-")
            elif "/" in gt_check and gt_check.split("/")[0] == gt_check.split("/")[1]:
                row.append(f"[green]{gt_check} pass[/green]")
            else:
                row.append(f"[red]{gt_check} FAIL[/red]")
        row.append(status)
        table.add_row(*row)

    console.print()
    console.print(table)
    console.print()

    fail = len(results) - ok
    if fail:
        console.print(f"[bold red]{fail}/{len(results)} datasets have errors.[/bold red]")
        sys.exit(1)
    else:
        console.print(f"[bold green]All {len(results)} datasets passed verification.[/bold green]")


if __name__ == "__main__":
    main()
