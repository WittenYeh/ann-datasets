# Makefile for Yandex Text-to-Image 10M dataset (T2I-10M).
# Source: BigANN Benchmarks (Yandex Research)
#   https://research.yandex.com/datasets/text-to-image-dataset-for-billion-scale-similarity-search
#
# Format: 200-d float32, query distribution differs from base:
#   base   = SE-ResNeXt-101 image embeddings
#   query  = DSSM text embeddings
# CANONICAL METRIC: inner product (cosine on the L2-normalised vectors).
# The artea bench harness is currently L2-only (default_context.hpp wires
# DistanceMetricsT::EUCLIDEAN); be aware that running with --metric L2
# will not match the official T2I-10M leaderboard. If you need IP, build
# an alternative type context or pre-normalise both query and base.
#
# Files:
#   * base.10M.fbin              — first 10M of the 1B base set, 8.0 GB
#   * query.public.100K.fbin     — 100K public queries (we keep first 10K
#                                  to match the published GT below)
#   * t2i.gt.10k.ibin (Zenodo)   — exact top-1000 ground truth for the
#                                  first 10K public queries
#                                  (paired with the RoarGraph paper,
#                                  https://zenodo.org/records/11090378).
#
# Yandex publishes a `GT_10M/text2image-10M` path in the big-ann-benchmarks
# datasets list, but it currently 404s on storage.yandexcloud.net, so we
# fall back to the Zenodo-hosted 10K-query GT until that path is restored.

DATASET_NAME := yandex-t2i-10m

BASE_URL     := https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I
GT_URL       := https://zenodo.org/records/11090378/files/t2i.gt.10k.ibin

# Number of queries we keep — must match the GT row count.
NUM_QUERIES  := 10000

# Downloaded source files
BASE_FBIN    := base.10M.fbin
QUERY_FBIN   := query.public.100K.fbin
GT_BIN       := t2i_groundtruth.10k.bin

# Final output files (.fvecs/.ivecs format)
BASE_FILE    := t2i10m_base.fvecs
QUERY_FILE   := t2i10m_query.fvecs
TRUTH_FILE   := t2i10m_groundtruth.ivecs

DOWNLOAD_FILES := $(BASE_FBIN) $(QUERY_FBIN) $(GT_BIN)
FINAL_FILES    := $(BASE_FILE) $(QUERY_FILE) $(TRUTH_FILE)

all: setup
.PHONY: all

setup: $(FINAL_FILES)
	@echo "-> $(DATASET_NAME) dataset setup complete!"
	@echo "   base : $(BASE_FILE) (10M × 200d, float32)"
	@echo "   query: $(QUERY_FILE) ($(NUM_QUERIES) × 200d, float32)"
	@echo "   gt   : $(TRUTH_FILE) ($(NUM_QUERIES) queries × top-1000 nbrs)"
	@echo "   metric: inner product (canonical) — see Makefile header."

# ── downloads ─────────────────────────────────────────────────────────────────
$(BASE_FBIN):
	@echo "-> Downloading T2I base.10M.fbin (~8 GB)..."
	wget --progress=bar:force:noscroll $(BASE_URL)/$(BASE_FBIN) -O $(BASE_FBIN)

$(QUERY_FBIN):
	@echo "-> Downloading T2I query.public.100K.fbin (~80 MB)..."
	wget --progress=bar:force:noscroll $(BASE_URL)/$(QUERY_FBIN) -O $(QUERY_FBIN)

$(GT_BIN):
	@echo "-> Downloading T2I 10K-query ground truth from Zenodo (~40 MB)..."
	wget --progress=bar:force:noscroll $(GT_URL) -O $(GT_BIN)

# ── conversions ───────────────────────────────────────────────────────────────
$(BASE_FILE): $(BASE_FBIN)
	@echo "-> Converting base vectors to fvecs format..."
	python3 ../fbin_to_fvecs.py $(BASE_FBIN) $(BASE_FILE)

# Trim queries to NUM_QUERIES so they line up with the published GT row count.
$(QUERY_FILE): $(QUERY_FBIN)
	@echo "-> Converting first $(NUM_QUERIES) query vectors to fvecs format..."
	python3 ../fbin_to_fvecs.py $(QUERY_FBIN) $(QUERY_FILE) --limit $(NUM_QUERIES)

$(TRUTH_FILE): $(GT_BIN)
	@echo "-> Converting ground truth to ivecs format..."
	python3 ../bin_to_ivecs.py $(GT_BIN) $(TRUTH_FILE)

fetch: $(DOWNLOAD_FILES)

clean:
	rm -f $(FINAL_FILES)

clean-all: clean
	rm -f $(DOWNLOAD_FILES)

.PHONY: all setup fetch clean clean-all
