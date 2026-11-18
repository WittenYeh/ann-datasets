# ANNDatasets documentation

Run the examples from the `ann-datasets` repository root.
For installation and a first dataset, start with [Quick Start](../README.md#quick-start).

## Dataset files

| Guide | SIFT-1M example | What it covers |
|-------|-----------------|----------------|
| [Base dataset](base-dataset.md) | `sift_base.fvecs` | The vectors to index, how the base is prepared, and how deduplication affects row IDs. |
| [Query dataset](query-sets.md) | `sift_query.fvecs` | The vectors to search with, when to use publisher queries, and how to generate queries with `num_queries` and `variance`. |
| [Ground truth](ground-truth.md) | `sift_groundtruth.ivecs` | The expected neighbor IDs for each query, the metric they use, and how to compute and check them. |

## Reference

| Guide | What it covers |
|-------|----------------|
| [Dataset catalog](datasets.md) | Choose a dataset by dimension, base size, query source, or modality and follow the link to its Makefile. |
| [File formats and conversion](file-formats.md) | Read and write vector files in Python, convert publisher formats, and extract a subset. |
| [Validation and tests](validation.md) | Inspect prepared files, check neighbor IDs, and run the correctness and Make integration tests. |
| [Distance profiling](distance-profiling.md) | Measure minimum and maximum L2 distances on a GPU, choose precision and tile size, and interpret aspect ratios. |

[Python downloads](https://www.python.org/downloads/) · [MIT license](https://opensource.org/licenses/MIT)
