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
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)

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

## Supported Datasets

| Directory | Dataset | Dim | # Base | # Query | Query Source | Type |
|-----------|---------|-----|--------|---------|--------------|------|
| `sift-1m` | SIFT-1M | 128 | 1M | 10K | Archive | Image |
| `gist-1m` | GIST-1M | 960 | 1M | 1K | Archive | Image |
| `deep-10m` | Deep-10M | 96 | 10M | 10K | Parent (Deep1B) | Image |
| `crawl` | Crawl | 300 | ~2M | 1K | **Generated** | Text |
| `msong` | MSONG | 420 | ~992K | 200 | Archive | Audio |
| `glove` | GloVe | 100 | ~1.2M | 1K | Archive | Text |
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
