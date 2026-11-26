# Ground truth

Ground truth records the correct neighbors for each query under the dataset's search metric.
With `Q` queries and `k` stored neighbors per query, the `.ivecs` file has `Q` rows of `k` integer IDs.
Each ID is a zero-based row number in the prepared base. [Vector file formats](file-formats.md#vector-files)

For SIFT-1M, row `i` in `sift_groundtruth.ivecs` contains the nearest neighbors of row `i` in `sift_query.fvecs`.
For example, if the first ID in that row is `42`, the closest vector is row `42` in `sift_base.fvecs`.
To evaluate recall@10, compare the ten IDs returned for the query with the first ten IDs in its ground-truth row.
Distinct base vectors can tie at the same distance.

## Where the neighbor IDs come from

| Recipe | Ground truth |
|--------|--------------|
| Recipes that deduplicate the base, including SIFT-1M | Compute exact L2 neighbors against the final base row order, with 1,000 neighbors per query by default. [Datasets with deduplicated bases](base-dataset.md#datasets-with-deduplicated-bases) |
| GloVe-200d and GloVe-300d | Compute exact L2 top-100 against the base prepared by the selected query method. |
| Deep1B and Yandex T2I-10M with `query-gen=published` | Both recipes convert supplied neighbor IDs to `.ivecs`, with T2I using inner product for the first 10,000 published queries. |

A ground-truth file belongs to a specific base, query order, and metric.
Changing the indexed rows or the queries requires matching neighbor IDs.
Changing only the queries lets you reuse the index built on the same base.
[Base dataset](base-dataset.md) · [Query dataset](query-sets.md)

## Building exact L2 ground truth with Make

The shared deduplication recipe provides a `groundtruth` target.
Make prepares the required base and queries before computing their neighbors. [Shared Make rules](../scripts/deduplicate.mk)

```bash
make -C sift-1m groundtruth
make -C sift-1m groundtruth GT_K=100 GT_CHUNK_SIZE=1000000
```

| Setting | Default | Effect |
|---------|--------:|--------|
| `GT_K` | 1,000 | Number of neighbor IDs per query, from 1 to the prepared base row count. |
| `GT_CHUNK_SIZE` | 2,000,000 | A lower limit on the number of base rows per search chunk reduces peak memory. |

Changing either setting rebuilds ground truth.
Make reuses up-to-date outputs when the inputs and settings match.
The shared recipe takes the thread count from `nproc` and launches the computation with `numactl --interleave=all`.

## Computing neighbors directly

The standalone utility reads `.fvecs` or `.bvecs` base and query files of the same dimension and searches with FAISS `IndexFlatL2` by default. Use `--metric ip` to rank by decreasing inner product.
For a base larger than `--chunk-size`, it searches one chunk at a time, converts each chunk's IDs to base row numbers, and merges the closest neighbors per query.
Its defaults are `--k 100` and `--chunk-size 2000000`. [Ground-truth utility](../scripts/compute_groundtruth.py)

```bash
OMP_NUM_THREADS="$(nproc)" numactl --interleave=all python3 scripts/compute_groundtruth.py \
    sift-1m/sift_base.fvecs sift-1m/sift_query.fvecs /tmp/sift_top100.ivecs \
    --k 100 --chunk-size 1000000
```

To check the resulting IDs against an exhaustive search on sampled queries, see [Verify sampled neighbors](validation.md#verify-sampled-neighbors).
