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

### SIFT Datasets
- **sift-10k**: Small SIFT dataset (10K vectors, 128 dimensions)
- **sift-1m**: SIFT 1 Million dataset (1M base vectors, 128 dimensions)
- **sift-1b**: SIFT 1 Billion dataset (1B base vectors, 128 dimensions)
  - **Warning**: Very large dataset (~120GB base file)
  - Downloads 4 separate files: base, learn, query, and ground truth
  - Use `make fetch` to download without decompression

### GIST Dataset
- **gist-1m**: GIST 1 Million dataset (1M vectors, 960 dimensions)

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

