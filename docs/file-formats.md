# File formats and conversion

## Vector files

In the TexMex formats, each row starts with a four-byte little-endian integer giving the number of elements that follow.
A base or query row contains coordinates.
A ground-truth row contains the IDs of the neighbors for one query, so its dimension is the number of neighbors.

| Extension | Element type | Bytes per element | Typical contents |
|-----------|--------------|------------------:|------------------|
| `.fvecs` | float32 | 4 | Base, query, or training vectors |
| `.bvecs` | uint8 | 1 | Byte-valued base or query vectors |
| `.ivecs` | int32 | 4 | Neighbor IDs |

For SIFT-1M, `sift_base.fvecs` is the indexed collection, `sift_query.fvecs` contains the queries, `sift_groundtruth.ivecs` contains their neighbors, and `sift_learn.fvecs` contains training vectors.
Each dataset's Makefile names its outputs. [Dataset catalog](datasets.md)

## Reading and writing in Python

`vecs_io.py` provides read-only memory maps for large files, contiguous arrays for full reads, and vectorized writers.
It follows the conventions used in deep1b_gt. [I/O module](../scripts/vecs_io.py) · [deep1b_gt](https://github.com/matsui528/deep1b_gt)

```python
from scripts.vecs_io import fvecs_mmap, fvecs_read, fvecs_write, ivecs_write, mmap_by_ext

base = fvecs_mmap("sift-1m/sift_base.fvecs")
queries = fvecs_read("sift-1m/sift_query.fvecs")
neighbors = mmap_by_ext("sift-1m/sift_groundtruth.ivecs")
print(base.shape, queries.shape, neighbors.shape)

fvecs_write("/tmp/first_queries.fvecs", queries[:10])
ivecs_write("/tmp/first_neighbors.ivecs", neighbors[:10])
```

`read_by_ext`, `mmap_by_ext`, and `write_by_ext` select the appropriate format from the filename extension.

## Converting publisher formats

Run these examples from the repository root with the corresponding source files available.
The binary vector converters can process a memory-mapped source in chunks.
`--limit` restricts the number of rows in the examples that use it.

| Source | Example command |
|--------|-----------------|
| BigANN float32 `.fbin` | `python3 scripts/fbin_to_fvecs.py deep-1b/base.1B.fbin /tmp/deep_sample.fvecs --limit 1000` |
| Signed int8 `.i8bin` | `python3 scripts/i8bin_to_fvecs.py spacev-10m/query.i8bin /tmp/spacev_query.fvecs` |
| Unsigned byte `.bvecs` | `python3 scripts/bvecs_to_fvecs.py bigann/bigann_query.bvecs /tmp/sift_query.fvecs` |
| BigANN ground-truth `.bin` | `python3 scripts/bin_to_ivecs.py deep-1b/deep_new_groundtruth.public.10K.bin /tmp/deep_gt.ivecs` |
| fastText text vectors with a header | `python3 scripts/vec_to_fvecs.py crawl/crawl-300d-2M.vec /tmp/crawl_sample.fvecs --max-vectors 1000` |
| GloVe text vectors | `python3 scripts/glove_to_fvecs.py glove-100d/glove100d.txt /tmp/glove.fvecs` |
| ANN-Benchmarks HDF5 | `python3 scripts/hdf5_to_fvecs.py glove-100-angular.hdf5 /tmp/glove100` |

The HDF5 converter uses `h5py` and maps the `train`, `test`, and `neighbors` arrays to base, query, and ground-truth files.
`--num-neighbors` controls how many stored neighbors to keep, up to 100 by default.

## Extracting a prefix

The subset utility copies the first `N` records in their original order.
This example writes a prefix of a BigANN base in the same format. [Subset utility](../scripts/extract_subset.py)

```bash
python3 scripts/extract_subset.py bigann/bigann_base.bvecs /tmp/sift_sample.bvecs 1000
```
