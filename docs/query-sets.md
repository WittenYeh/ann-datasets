# Query dataset

The query dataset contains the vectors submitted to the index during search.
If the base vectors have `D` coordinates, each of the `Q` query rows also has `D` coordinates.
In SIFT-1M, searching row `i` of `sift_query.fvecs` produces the results to compare with row `i` of the ground-truth file. [Ground truth](ground-truth.md)

## Published queries

When a publisher provides query vectors, the recipe uses those vectors.
A subset such as DEEP-10M reuses the queries from its parent.
Yandex T2I-10M uses the first 10,000 public queries to match the rows of its published ground truth.
The catalog identifies the source for each recipe. [Dataset catalog](datasets.md)

## Generated queries

Five datasets provide a base without a separate query set.
Their Makefiles call the same generator with two parameters.

| Dataset | Base used for sampling | Default `num_queries` |
|---------|------------------------|----------------------:|
| Crawl, GloVe-100d | Complete base after deduplication | 10,000 |
| YahooMusic | Complete base after deduplication | 1,000 |
| GloVe-200d, GloVe-300d | Complete converted base | 1,000 |

- **`num_queries`** is the number of distinct base row IDs sampled uniformly without replacement.
  It can range from 1 to the number of rows in the prepared base.
  The sampled rows keep their positions in the index.
- **`variance`** is the variance of the noise added independently to each coordinate, in squared units of the original coordinates.
  For a sampled row `x`, the query is `x + ε`, where each component of `ε` follows a zero-mean Gaussian.
  The default variance of `0.0001` gives a standard deviation of `0.01`.
  A variance of zero returns the original sampled vectors.

The generator uses a fixed seed of 42 and writes the perturbed coordinates directly as float32, including values outside the original coordinate range.
For the same base and query count, changing the variance keeps the same source rows. [Generator](../scripts/generate_queries.py) · [Make rules](../scripts/generate-queries.mk)

```bash
make -C crawl all num_queries=10000 variance=0.0001
make -C yahoomusic all num_queries=1000 variance=0.0004
```

## Rebuilds and older splits

Changing either parameter rebuilds the query file and its exact L2 ground truth.
Repeating the command with the same inputs and parameters reuses the outputs.
The chosen values are recorded in `*_query.fvecs.config`.

Older builds removed the sampled rows from the base.
The first `make all` with the current recipe restores the complete base from the source, applies the dataset's usual preprocessing, and regenerates queries and neighbors.
Rebuild indexes created from the old split because the restored rows change base IDs. [Base dataset](base-dataset.md)

## Standalone generation

To write a separate query set, pass the prepared `.fvecs` or `.bvecs` base and a new `.fvecs` output path.
The script reads the base through a read-only memory map.

```bash
python3 scripts/generate_queries.py crawl/crawl_base.fvecs /tmp/crawl_queries.fvecs \
    --num-queries 1000 --variance 0.0001
```

For an experiment that explicitly requires a held-out split, `extract_query.py` writes a smaller base and a separate query file. [Holdout utility](../scripts/extract_query.py)
