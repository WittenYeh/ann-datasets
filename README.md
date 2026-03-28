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

## Features

- **One-Command Setup**: Download and extract with `make all`
- **Auto-Skip Existing**: Smart checks prevent redundant downloads/extractions
- **Dataset Preview**: Built-in tool to inspect vector files
- **Multiple Formats**: Support for fvecs, bvecs, ivecs formats
- **Large Dataset Support**: Optimized for datasets up to 1B+ vectors

## Download and Decompress Target Dataset

Goto the subdirectory of target dataset you want to get.

For example, to get SIFT-1M dataset, run:

```bash
cd ./sift-1m
```

Then run:

```bash
make all
```

The download and extraction process includes:
- **Auto-skip existing files**: If files already exist, they won't be re-downloaded or re-extracted
- **Auto-extraction**: Archives are automatically extracted after download
- **Progress tracking**: Download progress is displayed with progress bars

## Inspect Dataset Information

After downloading a dataset, you can inspect its properties:

```bash
make info
```

This will display:
- Vector format (fvecs/bvecs/ivecs)
- Data type (float32/uint8/int32)
- Number of vectors
- Vector dimension
- File size

You can also use the `preview_dataset.py` tool directly:

```bash
python3 ../preview_dataset.py sift_base.fvecs
python3 ../preview_dataset.py sift_base.fvecs --preview 5  # Preview first 5 vectors
```

## Clean Target Dataset

-   `make clean`: Delete the extracted data files.
-   `make clean-all`: Delete both the extracted files and the original downloaded archive.

## Supported Datasets

| Directory | Dataset | Dim | # Base | # Query | Type | Homepage | Download URL |
|-----------|---------|-----|--------|---------|------|----------|--------------|
| `sift-10k` | SIFT-10K | 128 | 10K | 100 | Image | http://corpus-texmex.irisa.fr | `ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz` |
| `sift-1m` | SIFT-1M | 128 | 1M | 10K | Image | http://corpus-texmex.irisa.fr | `ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz` |
| `sift-10m` | SIFT-10M | 128 | 10M | 10K | Image | http://corpus-texmex.irisa.fr | Subset of SIFT1B (see `big-ann`) |
| `sift-100m` | SIFT-100M | 128 | 100M | 10K | Image | http://corpus-texmex.irisa.fr | Subset of SIFT1B (see `big-ann`) |
| `big-ann` | SIFT-1B | 128 | 1B | 10K | Image | http://corpus-texmex.irisa.fr | `http://corpus-texmex.irisa.fr/` |
| `gist-1m` | GIST-1M | 960 | 1M | 1K | Image | http://corpus-texmex.irisa.fr | `ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz` |
| `deep-1m` | Deep-1M | 96 | 1M | 10K | Image | https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search | `https://storage.yandexcloud.net/yandex-research/ann-datasets/deep1b/` |
| `msong` | MSONG | 420 | ~992K | - | Audio | http://www.ifs.tuwien.ac.at/mir/msd/ | `https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/msong.tar.gz` |
| `audio` | Audio | 192 | ~54K | - | Audio | https://www.cs.princeton.edu/cass/demos.htm | `https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/audio.tar.gz` |
| `glove` | GloVe | 100 | ~1.2M | - | Text | http://nlp.stanford.edu/projects/glove/ | `https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/glove1.2m.tar.gz` |
| `crawl` | Crawl | 300 | ~2M | - | Text | http://commoncrawl.org/ | `https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip` |

### Notes

- **SIFT / GIST**: Classic Texmex corpus datasets. Ready-to-use fvecs format.
- **Deep-1M**: CNN features from Yandex Deep1B, reduced by PCA and L2-normalized. Extracted as a 1M subset.
- **MSONG / Audio / GloVe**: Ready-to-use fvecs from [CUHK GQR](https://www.cse.cuhk.edu.hk/systems/hash/gqr/datasets.html). No groundtruth included; must be computed separately.
- **Crawl**: fastText Common Crawl 300d word vectors. Downloaded as text format (.vec) and auto-converted to fvecs. Query/groundtruth must be generated separately.
- **SIFT-10M / SIFT-100M / SIFT-1B**: Large-scale SIFT variants. 10M and 100M are auto-extracted from SIFT1B.

## Working with Large Datasets

### Extracting Subsets

For large datasets like SIFT1B and Deep1B, you can extract smaller subsets:

```bash
# Extract SIFT10M from SIFT1B (auto-downloads SIFT1B if needed)
cd sift-10m && make all

# Extract SIFT100M from SIFT1B (auto-downloads SIFT1B if needed)
cd sift-100m && make all

# Extract Deep1M from Deep1B (downloads and extracts automatically)
cd deep-1m && make setup
```

**Note**: SIFT10M and SIFT100M will automatically download and setup SIFT1B if it's not already present. The extraction process reuses the SIFT1B files without duplicating the large dataset.

The `extract_subset.py` tool can also be used directly:

```bash
python3 extract_subset.py bigann_base.bvecs sift10m_base.bvecs 10000000
```

### Dataset Dependencies

- **sift-10m** and **sift-100m** automatically initialize **big-ann** if needed
- **deep-1m** downloads Deep1B automatically and extracts the subset

## How to Add a New Dataset

1.  Create a new subdirectory in the root folder, e.g., `my-dataset`.
2.  Create a `Makefile` inside `my-dataset/`.
3.  Following the existing examples, define the dataset variables (`DATASET_NAME`, `ARCHIVE_FILE`, `URL`) and include the appropriate `recipe-*.mk` file from the root directory. Lastly, define the `all` target to trigger the `setup` process.

## Vector File Formats

This repository supports three vector file formats from the Texmex corpus:

- **fvecs**: Float32 vectors (4 bytes per element)
- **bvecs**: Uint8 vectors (1 byte per element)
- **ivecs**: Int32 vectors (4 bytes per element)

Each vector is stored with a 4-byte dimension header followed by the vector data.

