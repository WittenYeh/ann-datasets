# Base dataset

The base is the collection of vectors an index searches.
For a base with `N` rows and `D` coordinates per row, the index contains `N` vectors of dimension `D`.
A vector's ID is its zero-based row number in the prepared base file.
For SIFT-1M, build the index on `sift-1m/sift_base.fvecs`.

## Where the base comes from

A recipe can unpack the publisher's vectors, convert them to a supported file format, or take a subset of a larger collection.
For example, DEEP-10M takes the first 10 million rows from Deep1B before deduplication.
The catalog gives the nominal size and links to each dataset's Makefile. [Dataset catalog](datasets.md)

## Datasets with deduplicated bases

SIFT-1M, GIST-1M, Crawl, GloVe-100d, YahooMusic, Tiny-5M, DEEP-10M, and SpaceV-10M use the same deduplication recipe.
It reads `*_base.raw.fvecs` and writes `*_base.fvecs`, keeping the bytes and input order of the first occurrence of each distinct vector.
With `query-gen=drawout`, the raw input to deduplication contains the rows left after sampling. [Query methods](query-sets.md)
For SIFT-1M, the following target prepares the base and its report. [Shared Make rules](../scripts/deduplicate.mk)

```bash
make -C sift-1m deduplicate
```

The report gives the actual indexed row count, which can be smaller than the size in the dataset name.
If you replace or reorder the base, rebuild its index and use neighbor IDs computed for the new rows. [Ground truth](ground-truth.md)

On Linux the recipe sets the OpenMP, Numba, Polars, and BLAS thread counts from `nproc` and launches deduplication with `numactl --interleave=all`.
Install the Python dependencies from [requirements.txt](../requirements.txt).

## Deduplication and its report

To write a separate deduplicated base and JSON report, run the utility on the raw base.

```bash
OMP_NUM_THREADS="$(nproc)" NUMBA_NUM_THREADS="$(nproc)" POLARS_MAX_THREADS="$(nproc)" \
    numactl --interleave=all python3 scripts/deduplicate.py \
    sift-1m/sift_base.raw.fvecs /tmp/sift_unique.fvecs \
    --threads "$(nproc)" --report /tmp/sift_dedup.json
```

The script supports `.fvecs` and `.bvecs`.
It checks dimension headers, record lengths, and finite float coordinates in bounded NumPy chunks.
Polars treats each complete vector as an array element.
The code sorts the first-occurrence IDs to retain input order.
Float equality is exact, including `+0.0 == -0.0`.

```python
ids = pl.Series("vector", vectors).arg_unique().to_numpy()
ids = np.sort(ids)
```

Numba copies the selected records into a temporary memory mapping.
Flushing the mapping and atomically renaming the file publishes the result.
`--threads` sets the copy workers.
`POLARS_MAX_THREADS` sets the Polars pool at startup.
Individual operations may use fewer workers.
Allow disk space for both vector files and up to O(number of vectors × dimensions) additional memory for Polars.

The Make recipe writes `*_base.fvecs.dedup.json`.
It contains input, unique, and removed row counts, the `polars` backend, and the copy and Polars thread counts.
`validation_seconds` and `deduplicate_seconds` cover validation and row selection.
`copy_seconds` and `flush_seconds` measure separate parts of `write_seconds`, which also includes file setup and publication.
Copy time includes Numba compilation or cache loading when applicable.
Earlier reports used the fields `hash_seconds`, `compare_seconds`, and `hash_algorithm`.
[Deduplication utility](../scripts/deduplicate.py)
