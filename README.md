# ANNDatasets

<!-- pyfiglet -f ansi_shadow -w 120 "ANN Datasets" -->
```text
 █████╗ ███╗   ██╗███╗   ██╗    ██████╗  █████╗ ████████╗ █████╗ ███████╗███████╗████████╗███████╗
██╔══██╗████╗  ██║████╗  ██║    ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██╔════╝██╔════╝╚══██╔══╝██╔════╝
███████║██╔██╗ ██║██╔██╗ ██║    ██║  ██║███████║   ██║   ███████║███████╗█████╗     ██║   ███████╗
██╔══██║██║╚██╗██║██║╚██╗██║    ██║  ██║██╔══██║   ██║   ██╔══██║╚════██║██╔══╝     ██║   ╚════██║
██║  ██║██║ ╚████║██║ ╚████║    ██████╔╝██║  ██║   ██║   ██║  ██║███████║███████╗   ██║   ███████║
╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═══╝    ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝   ╚═╝   ╚══════╝
```

Prepare base vectors, queries, and ground-truth neighbor IDs for approximate nearest neighbor benchmarks.

## Quick Start

Run these commands from the `ann-datasets` directory on Linux.
The SIFT-1M recipe uses Python 3.10+, GNU Make 4.3+, `numactl`, `wget`, and `tar`.
The Crawl example below also uses `unzip`.

```bash
python3 -m pip install -r requirements.txt
make -C sift-1m all
make -C sift-1m info
```

The prepared files are in `sift-1m/`.
The `info` target prints their row counts and dimensions.

To reclaim disk space, `make -C sift-1m clean` removes the prepared files.
`make -C sift-1m clean-all` also removes the downloaded archive.

## Base sets

Build the index on `sift_base.fvecs`.
SIFT-1M keeps the first occurrence of each distinct vector.
Its zero-based row number in this file is the ID returned by the index. [Base dataset](docs/base-dataset.md)

## Query sets

Search the index with the vectors in `sift_query.fvecs`, one query per row.
SIFT-1M uses the publisher's queries.
For a dataset such as Crawl, Make samples base rows and adds Gaussian noise.
The sampled rows stay in the base.
[Query sources and parameters](docs/query-sets.md)

```bash
make -C crawl all num_queries=10000 variance=0.0001
```

## Ground truth

SIFT-1M computes exact L2 neighbors on its prepared base.
Row `i` of `sift_groundtruth.ivecs` lists the base IDs for query `i`, closest first.
Compare the IDs your index returns with this row to measure recall.
[Ground truth](docs/ground-truth.md)

## Documentation

[Documentation index](docs/README.md) · [Dataset catalog](docs/datasets.md)
