# Distance profiling

## Minimum and maximum pair distances

The GPU profiler examines every pair of distinct rows in a float32 `.fvecs` base, using the coordinates as stored.
It reports the smallest and largest Euclidean distances, the two pairs of row IDs, and the aspect ratio `delta = max_distance / min_distance`.
It requires NumPy, CUDA-enabled PyTorch, and a GPU with space for the full base. [Pair-distance profiler](../scripts/profile_pair_distance.py)

```bash
OMP_NUM_THREADS="$(nproc)" numactl --interleave=all python3 scripts/profile_pair_distance.py \
    --base sift-1m/sift_base.fvecs --device cuda:0 --threads "$(nproc)" \
    --output /tmp/sift-pair-distance.json
```

When this repository is checked out inside `artea-benchmark`, `--dataset` resolves a base from the parent project's `configs/datasets.json`.

```bash
OMP_NUM_THREADS="$(nproc)" numactl --interleave=all python3 scripts/profile_pair_distance.py \
    --dataset deep-10m --device cuda:0 --threads "$(nproc)" \
    --matmul float32 --compile --output /tmp/deep-pair-distance.json
```

## Precision, memory, and output

The default uses float64 matrix multiplication.
`--matmul float32` and `--matmul tf32` screen tiles with a conservative rounding-error bound, then refine candidate extrema in float64.
For small integer coordinates such as SIFT or SpaceV, the profiler checks whether the dot products are exactly representable.
It independently recomputes the two reported pairs on the CPU.

For `N` vectors of dimension `D`, the exhaustive floating-point scan costs O(N²D) work.
GPU memory is O(ND + block_size²).
The base stays resident in float32 and the active tiles are widened as needed.
`--block-size` sets the tile size and defaults to 8192.
Optional `--compile` fuses distance expansion and reductions and requires PyTorch 2+, Triton, a working C/C++ compiler, and Python development headers.

The JSON report contains the input shape, distances and squared distances, `min_pair_ids` and `max_pair_ids` (zero-based input rows), the pair count, precision settings, and elapsed time.
The scan continues when a pair has distance zero.

| Condition | `delta` | `delta_status` |
|-----------|---------|----------------|
| Positive minimum distance | Maximum divided by minimum | `finite` |
| Zero minimum, positive maximum | `null` | `infinite` |
| All vectors coincide | `null` | `undefined` |

## Sampling and duplicate locations

When the question concerns the closest *distinct locations*, `profile_aspect_ratio.py` also reports the smallest strictly positive distance.
`--max-points` selects a reproducible subsample.
Its extrema describe that sample. [Aspect-ratio utility](../scripts/profile_aspect_ratio.py)

```bash
OMP_NUM_THREADS="$(nproc)" numactl --interleave=all python3 scripts/profile_aspect_ratio.py \
    --base sift-1m/sift_base.fvecs --max-points 10000 --double \
    --output /tmp/sift-sample-aspect.json
```

To run the profiler's correctness tests, use `python3 -m unittest discover -s tests -p test_profile_pair_distance.py -v`. [Validation and tests](validation.md)
