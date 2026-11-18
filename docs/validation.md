# Validation and tests

## Inspect a prepared file

For SIFT-1M, `make -C sift-1m info` prints a summary of the prepared files.
The preview utility works on individual `.fvecs`, `.bvecs`, and `.ivecs` files.
Add `--preview` to see values from the first few rows. [Preview utility](../scripts/preview_dataset.py)

```bash
python3 scripts/preview_dataset.py sift-1m/sift_query.fvecs --preview 5
```

## Check downloaded datasets

The checker discovers dataset directories containing `*_base.fvecs` or `*_base.bvecs`.
It looks for matching `*_query` and `*_groundtruth.ivecs` files, checks record sizes and dimensions, compares the query and ground-truth row counts, and checks that neighbor IDs are within the base.
It uses NumPy and Rich. [Integrity checker](../scripts/verify_datasets.py)

```bash
python3 scripts/verify_datasets.py
```

A `WARN` row can mean a companion file has a different name.
For example, Deep1B uses `deep1B_queries.fvecs`, and BigANN uses `idx_1000M.ivecs`.
Consult the recipe for the filenames and inspect those files directly.

## Verify sampled neighbors

In a checkout of float32 L2 datasets, add `--verify-gt` to recompute neighbors with FAISS for a fixed-seed sample of queries in each discovered dataset.
The default sample size is 20.
The checker compares sets of IDs.
Equal-distance ties at the top-k boundary can produce different valid sets.
The sampled check reads `.fvecs` base and query inputs and uses L2.
Yandex T2I's published ground truth uses inner product.

```bash
OMP_NUM_THREADS="$(nproc)" numactl --interleave=all python3 scripts/verify_datasets.py \
    --verify-gt --gt-samples 20
```

## Run the test suite

Install the repository dependencies, including `faiss-cpu` for the ground-truth and Make tests.
The integration tests also use GNU Make 4.3+, `numactl`, and `unzip`.
The distance-profiling tests use optional PyTorch and CUDA.

```bash
python3 -m pip install -r requirements.txt
OMP_NUM_THREADS="$(nproc)" NUMBA_NUM_THREADS="$(nproc)" POLARS_MAX_THREADS="$(nproc)" \
    numactl --interleave=all python3 -m unittest discover -s tests -v
```

- Deduplication tests compare complete files with an independent first-occurrence reference, including row order, byte preservation, signed zeros, and float32 edge cases.
- Ground-truth tests compare valid, distinct IDs and their rank distances against exhaustive float64 calculations.
  Cases include top-1000, tied boundary distances, and chunks smaller than k.
- Make integration tests exercise the eight deduplicated recipes and GloVe-200d/300d on small local inputs.
  They cover published and generated queries, text conversion, older splits, parameter changes, and reuse of prepared files.
- Generator tests check reproducibility, the noise mean and covariance, and preservation of the base.
  Simulated write and publication failures check that previous outputs survive and temporary files are removed.
