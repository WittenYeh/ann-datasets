# Dataset catalog

The directory links open the Makefiles that prepare each dataset.
Shared Make rules and Python utilities live in the [scripts directory](../scripts/).
Base sizes are nominal or rounded unless an exact count is shown.
Deduplication can reduce the indexed row count.
Query counts are the defaults used by the recipes.

| Directory | Dimensions | Base rows | Queries | Query source | Data |
|-----------|-----------:|----------:|--------:|--------------|------|
| [sift-1m](../sift-1m/Makefile) | 128 | 1M | 10,000 | Publisher | SIFT image descriptors |
| [gist-1m](../gist-1m/Makefile) | 960 | 1M | 1,000 | Publisher | GIST image descriptors |
| [bigann](../bigann/Makefile) | 128 | 1B | 10,000 | Publisher | SIFT1B |
| [bigann-10m](../bigann-10m/Makefile) | 128 | 10M | 10,000 | Parent (SIFT1B) | SIFT, converted to float32 |
| [bigann-100m](../bigann-100m/Makefile) | 128 | 100M | 10,000 | Parent (SIFT1B) | SIFT, uint8 |
| [deep-1b](../deep-1b/Makefile) | 96 | 1B | 10,000 | Publisher | DEEP image embeddings |
| [deep-1m](../deep-1m/Makefile) | 96 | 1M | 10,000 | Parent (Deep1B) | DEEP image embeddings |
| [deep-10m](../deep-10m/Makefile) | 96 | 10M | 10,000 | Parent (Deep1B) | DEEP image embeddings |
| [deep-100m](../deep-100m/Makefile) | 96 | 100M | 10,000 | Parent (Deep1B) | DEEP image embeddings |
| [tiny5m](../tiny5m/Makefile) | 384 | 5M | 1,000 | Publisher | Image |
| [imagenet](../imagenet/Makefile) | 150 | ~2.3M | 200 | Publisher | Image |
| [ukbench](../ukbench/Makefile) | 128 | ~1.1M | 200 | Publisher | Image |
| [cifar60k](../cifar60k/Makefile) | 512 | 60K | As shipped | Publisher | Image |
| [nuswide](../nuswide/Makefile) | 500 | ~269K | 200 | Publisher | Image |
| [yandex-t2i-10m](../yandex-t2i-10m/Makefile) | 200 | 10M | 10,000 | Publisher, first 10K | Cross-modal image base and text queries |
| [spacev-10m](../spacev-10m/Makefile) | 100 | 10M | 29,316 | Publisher | Web-search embeddings |
| [crawl](../crawl/Makefile) | 300 | ~2M | 10,000 | Generated | fastText Common Crawl |
| [glove-100d](../glove-100d/Makefile) | 100 | ~1.3M | 10,000 | Generated | GloVe 2024 |
| [glove-200d](../glove-200d/Makefile) | 200 | ~1.3M | 1,000 | Generated | GloVe 2024 |
| [glove-300d](../glove-300d/Makefile) | 300 | ~1.3M | 1,000 | Generated | GloVe 2024 |
| [word2vec](../word2vec/Makefile) | 300 | 1M | As shipped | Publisher | Word embeddings |
| [audio](../audio/Makefile) | 192 | 53,387 | 200 | Publisher | Audio |
| [msong](../msong/Makefile) | 420 | 994,185 | 1,000 | Publisher | Million Song audio features |
| [yahoomusic](../yahoomusic/Makefile) | 300 | ~1.8M | 1,000 | Generated | Music embeddings |

**Parent** recipes take a prefix of the parent base and reuse its published queries.
Preparing one of these subsets can also download and prepare its billion-scale parent.
**Generated** recipes sample the prepared base and perturb the sampled vectors. [Query generation and defaults](query-sets.md)

**Yandex T2I-10M** uses inner product and keeps the first 10,000 public text queries covered by its ground-truth file.
**SpaceV-10M** uses L2 and the Standard Track's in-distribution document queries.
Its signed int8 vectors are converted to float32.
