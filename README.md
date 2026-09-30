# ANNDatasets

```
    _    _   _ _   _   ____        _                  _
   / \  | \ | | \ | | |  _ \  __ _| |_ __ _ ___  ___| |_ ___
  / _ \ |  \| |  \| | | | | |/ _` | __/ _` / __|/ _ \ __/ __|
 / ___ \| |\  | |\  | | |_| | (_| | || (_| \__ \  __/ |_\__ \
/_/   \_\_| \_|_| \_| |____/ \__,_|\__\__,_|___/\___|\__|___/
```

**Easy-to-use ANN benchmark datasets with a single MAKE command.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Quick Start

```bash
# Download and setup a dataset
cd sift-1m
make all

# Inspect dataset information
make info

# Clean up
make clean      # Remove extracted files
make clean-all  # Remove everything including archives
```

### Deduplicated benchmark datasets

For **SIFT-1M, GIST-1M, Crawl, GloVe-100d, YahooMusic, Tiny-5M, DEEP-10M,
and SpaceV-10M**, `make all` performs this pipeline:

1. Download/convert the original base and the publisher's queries, when provided.
   Crawl, GloVe-100d and YahooMusic instead keep the existing seed-42 query split
   (10,000, 10,000 and 1,000 queries, respectively), before deduplication.
2. Keep the original benchmark base in `*_base.raw.fvecs`. Deduplicate it into
   the usual `*_base.fvecs` filename, keeping the first occurrence in input order.
3. Compute our own **top-1000 exact L2 neighbors** against the deduplicated base.
   Official ground truth is neither converted nor used. Archives that bundle GT
   with vectors are downloaded as usual, but their GT is not extracted.

Install NumPy, Polars, Numba and `faiss-cpu` into the Python environment used by Make:
`python3 -m pip install numpy 'polars>=1.0' numba faiss-cpu`. This pipeline also
requires Linux, GNU Make 4.3+ and `numactl`.
Make configures the Polars, Numba and OpenMP thread pools using `nproc`, with
`numactl --interleave=all` from process startup.

```bash
make -C sift-1m all                        # deduplicate + fresh top-1000 GT
make -C sift-1m deduplicate                # only prepare the deduplicated base
make -C sift-1m groundtruth GT_K=1000       # compute GT if stale or missing
make -C sift-1m all GT_K=100 GT_CHUNK_SIZE=1000000
```

An unchanged `make all` reuses both outputs. Changing `GT_K` or `GT_CHUNK_SIZE`
rebuilds GT. Each base also has a `*.dedup.json` report with counts and timings.
`make clean` removes generated files, including the raw base and reports;
`make clean-all` additionally removes downloaded sources.

**IDs are row numbers in the new base.** After deduplication, rebuild indexes
and use the newly computed GT together with that base. Dataset names retain
their nominal sizes, while the actual row count can shrink. Query contents are
preserved; learn/training vectors are not deduplicated. Distinct vectors can
still be at the same distance from a query: removing identical base vectors
does not eliminate all distance ties.

## Supported Datasets

| Directory | Dataset | Dim | # Base | # Query | Query Source | Type |
|-----------|---------|-----|--------|---------|--------------|------|
| `sift-1m` | SIFT-1M | 128 | 1M | 10K | Archive | Image |
| `gist-1m` | GIST-1M | 960 | 1M | 1K | Archive | Image |
| `deep-10m` | Deep-10M | 96 | 10M | 10K | Parent (Deep1B) | Image |
| `crawl` | Crawl | 300 | ~2M | 10K | **Generated** | Text |
| `msong` | MSONG | 420 | ~992K | 200 | Archive | Audio |
| `glove` | GloVe | 100 | ~1.2M | 1K | Archive | Text |
| `glove-100d` | GloVe 2024 | 100 | ~1.28M | 10K | **Generated** | Text |
| `imagenet` | ImageNet | 150 | ~2.3M | 200 | Archive | Image |
| `ukbench` | UKBench | 128 | ~1.1M | 200 | Archive | Image |
| `yahoomusic` | Yahoo Music | 300 | ~1.8M | 1K | **Generated** | Latent |
| `tiny5m` | Tiny-5M | 384 | 5M | 1K | Archive | Image |
| `yandex-t2i-10m` | Yandex T2I-10M | 200 | 10M | 10K | Archive (truncated) | Cross-modal (image base, text query, **inner product**) |
| `spacev-10m` | Microsoft SpaceV-10M | 100 | 10M | 29,316 | Archive (in-distribution) | Web-search (int8 → float32, L2, **Standard Track**) |

**Query Source** column:
- **Archive**: Query vectors are provided in the original download archive by the dataset publisher.
- **Parent**: Query vectors are copied from the parent dataset (e.g., Deep1B).
- **Generated**: No query vectors in the original archive. We randomly sample vectors from the base set as queries and remove them from the base to ensure no overlap.

## Utility Scripts

### profile_pair_distance.py — GPU Pair-Distance Profile

Requires NumPy and CUDA-enabled PyTorch. Scan the **complete base** to obtain
the minimum and maximum Euclidean distances between distinct row IDs, and
`delta = max_distance / min_distance`. Coordinates are not normalized or sampled.

```bash
python3 profile_pair_distance.py --base sift-1m/sift_base.fvecs \
    --device cuda:0 --output /tmp/sift-pair-distance.json
# With this repository inside artea-benchmark, resolve the base via its config:
python3 profile_pair_distance.py --dataset deep-10m --device cuda:0 \
    --matmul float32 --compile --output /tmp/deep-pair-distance.json
```

The default uses float64. Optional `--matmul float32` or `--matmul tf32` screens
tiles with a conservative rounding-error bound, then recomputes possible
extrema in float64. TF32 can be particularly useful for small integer datasets
such as SIFT and SpaceV: the script checks when their dot products are exactly
representable. Both reported pairs are also recomputed independently on the CPU.
`--compile` fuses distance expansion and reductions; it requires PyTorch 2+,
Triton and a working C/C++ compiler with Python development headers.

This is exhaustive floating-point search, not approximate nearest-neighbor
search or arbitrary-precision arithmetic. It takes O(N²D) work and
O(ND + block_size²) GPU memory. Use `--block-size` to adjust temporary memory
(default 8192); the full float32 vector array must also fit on the selected GPU.
Float64 GEMM widens only the active tiles, keeping the resident data compact.

JSON reports include distances and their squares, `min_pair_ids`, `max_pair_ids`
(zero-based rows of the input file), `delta`, input shape, evaluated pair count,
precision settings and elapsed time. A zero minimum does **not** end the scan.
For a zero minimum, `delta` is `null` and `delta_status` is `infinite` when the
maximum is positive, or `undefined` when all vectors coincide (0/0).

```bash
python3 -m unittest discover -s tests -p test_profile_pair_distance.py -v
```

### deduplicate.py — Stable Exact Vector Deduplication

```bash
OMP_NUM_THREADS=$(nproc) NUMBA_NUM_THREADS=$(nproc) POLARS_MAX_THREADS=$(nproc) \
    numactl --interleave=all python3 deduplicate.py \
    sift-1m/sift_base.raw.fvecs sift-1m/sift_base.fvecs \
    --threads "$(nproc)" --report sift-1m/sift_base.fvecs.dedup.json
```

The script memory-maps the input and validates dimension headers and finite
coordinates in bounded NumPy chunks. Polars treats each complete vector as an
`Array` element and returns first-occurrence IDs:

```python
ids = pl.Series("vector", vectors).arg_unique().to_numpy()
ids = np.sort(ids)
```

Numba copies those original records in parallel into a temporary output mapping.
After flushing that mapping, an atomic rename publishes the completed file.
`--threads` controls output-copy workers. Polars uses its process-wide pool,
configured with `POLARS_MAX_THREADS` before launch; individual operations may
use fewer workers. Numba compiles and caches only the output-copy kernel.

The JSON report records `backend: "polars"`, `threads` (copy workers),
`polars_threads`, `validation_seconds` and `deduplicate_seconds`. The old
`hash_seconds`, `compare_seconds` and `hash_algorithm` fields are removed.
`copy_seconds` and `flush_seconds` are separate; `write_seconds` includes both,
file setup and publication. Copy timings include JIT compilation/cache loading
when it occurs; flushing depends on filesystem throughput.

Both `fvecs` and `bvecs` are supported. Float equality is exact, without any
distance calculation or tolerance; `+0.0` and `-0.0` compare equal. The first
occurrence's bytes are preserved. Non-finite floats, inconsistent dimension
headers, truncated records and empty inputs are rejected. The input mapping
is backed by the file. Polars may copy coordinates and allocate temporary arrays,
so allow O(number of vectors x dimensions) additional memory. The output is a
separate file, so allow disk space for both the raw and deduplicated base.
The utility requires `numpy`, `polars>=1.0` and `numba`;
`faiss-cpu` is needed by the subsequent ground-truth computation.

Run correctness and Make integration tests (including all eight pipelines):

```bash
python3 -m pip install -r requirements.txt
OMP_NUM_THREADS=$(nproc) NUMBA_NUM_THREADS=$(nproc) POLARS_MAX_THREADS=$(nproc) numactl --interleave=all \
    python3 -m unittest discover -s tests -v
```

FAISS is required for the complete suite: a missing installation fails the
tests instead of silently skipping GT/Make checks. The suite compares complete
deduplicated files against an independent first-occurrence reference, including
byte preservation, output order, signed zeros and float32 edge cases. All eight
Make pipelines run on small local fixtures, including Crawl/GloVe text conversion
and query extraction. GT checks use independent float64 exhaustive distances,
require distinct valid IDs and sorted nearest-neighbor ranks, and cover top-1000,
equal-distance boundaries and chunks smaller than k. Injected write/publication
failures check that previous data and GT files survive and temporary files are
removed. These are correctness checks, not large-dataset performance or crash/
power-loss durability tests.

### vecs_io.py — Shared I/O Module

Memory-mapped and vectorized read/write for fvecs/bvecs/ivecs files, aligned with the conventions from [deep1b_gt](https://github.com/matsui528/deep1b_gt).

```python
from vecs_io import fvecs_mmap, fvecs_read, fvecs_write, ivecs_write

# Memory-mapped read (zero-copy, ideal for large files)
xb = fvecs_mmap("sift_base.fvecs")        # returns read-only (n, d) view
xq = fvecs_read("sift_query.fvecs")       # returns contiguous (n, d) array

# Vectorized write (no Python loops)
fvecs_write("output.fvecs", data)          # data: (n, d) float32 array
ivecs_write("output.ivecs", ids)           # ids: (n, k) int32 array

# Auto-select by file extension
from vecs_io import mmap_by_ext, read_by_ext, write_by_ext
data = mmap_by_ext("sift_base.fvecs")      # auto-detects fvecs/bvecs/ivecs
```

### compute_groundtruth.py — Exact k-NN Ground Truth

Computes exact ground truth using FAISS brute-force search. For large datasets, automatically switches to chunked search to bound memory usage.

```bash
# Basic usage
python3 compute_groundtruth.py base.fvecs query.fvecs groundtruth.ivecs --k 100

# Control memory via chunk size (default: 2M vectors per chunk)
python3 compute_groundtruth.py deep10m_base.fvecs deep10m_query.fvecs gt.ivecs --chunk-size 1000000
```

Requires `faiss-cpu`: `pip install faiss-cpu`

### extract_query.py — Random Query Extraction

Randomly samples query vectors from a base file, removes them from the base set, and optionally computes ground truth — all in one step.

```bash
# Extract 1000 random queries (seed=42), write compacted base and query files
python3 extract_query.py input_base.fvecs base_out.fvecs query_out.fvecs --num-queries 1000 --seed 42

# Also compute ground truth in the same step
python3 extract_query.py input_base.fvecs base_out.fvecs query_out.fvecs --gt gt_out.ivecs --k 100
```

The extraction guarantees:
- **Random sampling**: `np.random.default_rng(seed).choice(n, nq, replace=False)`
- **No overlap**: Sampled query vectors are removed from the output base file
- **Reproducible**: Fixed seed (default 42) ensures identical splits across runs
- **Memory-efficient**: Uses memmap + chunked writing for large files

### extract_subset.py — Subset Extraction

Extracts the first N vectors from a large vector file using memory-mapped I/O.

```bash
python3 extract_subset.py bigann_base.bvecs sift10m_base.bvecs 10000000
```

### fbin_to_fvecs.py — Format Conversion

Converts BigANN `.fbin` format to `.fvecs` using memory-mapped chunked I/O (safe for billion-scale files).

```bash
python3 fbin_to_fvecs.py base.1B.fbin deep1B_base.fvecs
```

### vec_to_fvecs.py — Text to Binary Conversion

Converts fastText `.vec` text format to `.fvecs` binary format with streaming chunked output.

```bash
python3 vec_to_fvecs.py crawl-300d-2M.vec crawl_base.fvecs
```

### hdf5_to_fvecs.py — HDF5 Conversion

Converts ann-benchmarks.com HDF5 format to fvecs/ivecs.

```bash
python3 hdf5_to_fvecs.py glove-100-angular.hdf5 glove100
```

### preview_dataset.py — Dataset Inspector

```bash
python3 preview_dataset.py sift_base.fvecs
python3 preview_dataset.py sift_base.fvecs --preview 5  # Preview first 5 vectors
```

### verify_datasets.py — Dataset Integrity Checker

Auto-discovers all downloaded datasets and verifies file existence, format integrity, dimension consistency, and ground truth validity. Requires `rich` for table output.

```bash
# Basic verification (file checks only)
python3 verify_datasets.py

# Also brute-force verify ground truth correctness on sampled queries
python3 verify_datasets.py --verify-gt --gt-samples 20
```

Checks performed:
- **File existence**: base, query, and ground truth files present
- **Format integrity**: valid fvecs/bvecs/ivecs format, no truncation
- **Dimension consistency**: base dim == query dim
- **GT validity**: row count matches query count, IDs within base range
- **GT correctness** (with `--verify-gt`): brute-force k-NN on sampled queries matches stored GT

Requires `pip install rich` and optionally `pip install faiss-cpu` for `--verify-gt`.

## Vector File Formats

This repository uses three vector file formats from the Texmex corpus:

- **fvecs**: Float32 vectors (4 bytes per element)
- **bvecs**: Uint8 vectors (1 byte per element)
- **ivecs**: Int32 vectors (4 bytes per element)

Each vector is stored with a 4-byte dimension header followed by the vector data.
