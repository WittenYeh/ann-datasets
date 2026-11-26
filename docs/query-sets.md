# Choosing a query set

Set `query-gen` on the dataset's Make command. Run the examples from the `ann-datasets` directory.

| Method | Query vectors | Indexed base |
|--------|---------------|--------------|
| `drawout` | Sample original rows uniformly without replacement, using seed 42. Write them unchanged in ascending source-row order. | Remove the sampled rows, then apply the dataset's usual deduplication, if any. |
| `gaussian` | Sample the prepared base uniformly without replacement, using seed 42. Add independent zero-mean Gaussian noise to each coordinate and write float32 queries. | Keep the complete prepared base, including the sampled rows. |
| `published` | Use the query file supplied by the publisher or inherited from the parent dataset. | Use the dataset's published recipe. |

## Datasets that generate their own queries

Crawl, GloVe-100d, GloVe-200d, GloVe-300d, and YahooMusic default to `drawout` and support both generated methods.
Crawl and GloVe-100d default to 10,000 queries. The other three default to 1,000.

```bash
make -C crawl all
make -C crawl all query-gen=drawout num_queries=10000
make -C crawl all query-gen=gaussian num_queries=10000 variance=0.0001
```

The first two commands produce the same held-out split. With the default count, `drawout` reproduces the historical seed-42 split.
Sampling selects row IDs, so duplicate vectors on different source rows can appear in both sets.
Crawl, GloVe-100d, and YahooMusic deduplicate the remaining rows after drawing them out.
In Gaussian mode, these three deduplicate the full source before sampling. GloVe-200d and GloVe-300d use the converted vectors directly.

- `num_queries` counts distinct source row IDs. Drawout requires at least one query and at least one remaining base row. Gaussian accepts 1 through the size of the prepared base. The final base must also contain enough rows for the requested ground-truth top-k.
- `variance` applies only to `gaussian`. The default per-coordinate variance of `0.0001` gives a standard deviation of `0.01` in the original units. At zero variance, each query is an exact copy of a row that stays in the base. Changing only the variance keeps the same source row IDs. [Gaussian generator](../scripts/generate_queries.py) · [Drawout script](../scripts/extract_query.py)

These five recipes write the selected base, query, and exact L2 ground truth to the usual dataset filenames.
Make records the method and effective parameters in `*_query.fvecs.config`. A build with older or missing configuration regenerates the affected files from the complete source.
Repeating the same build reuses its outputs. Changing the method or the drawout count prepares a new base and ground truth. Rebuild any index that uses the old base IDs. Changing just the Gaussian count or variance rebuilds queries and ground truth and lets you reuse the index. [Shared Make rules](../scripts/generate-queries.mk)

## Datasets with published queries

All other catalog entries default to `published`. For example, DEEP-10M reuses the queries of Deep1B, and Yandex T2I-10M takes the first 10,000 public text queries covered by its published ground truth. [Dataset catalog](datasets.md)

```bash
make -C sift-1m all query-gen=published
make -C sift-1m all query-gen=drawout num_queries=1000
make -C sift-1m all query-gen=gaussian num_queries=1000 variance=0.0001
```

The generated versions are in `sift-1m/variants/drawout/` and `sift-1m/variants/gaussian/`. Use the base, query, and `groundtruth.ivecs` from the selected directory together. The base keeps its source format. Drawout queries use that format and Gaussian queries use `.fvecs`. The Gaussian base links to the prepared publisher base.

Omitting `num_queries` uses the published query count. Each generated version computes its own exact neighbors on the selected base. Yandex T2I uses inner product, and sampling its image base produces an image-to-image experiment. Use `published` for the original text-to-image workload. [Ground truth](ground-truth.md)

## Standalone generation

For separate output paths, invoke either script directly. The drawout script accepts the complete source and writes both the reduced base and the held-out queries. The Gaussian script accepts the prepared base and writes the queries.

```bash
python3 scripts/extract_query.py crawl/crawl_base_orig.fvecs /tmp/base.fvecs /tmp/heldout.fvecs --num-queries 1000
python3 scripts/generate_queries.py crawl/crawl_base.fvecs /tmp/noisy.fvecs --num-queries 1000 --variance 0.0001
```
