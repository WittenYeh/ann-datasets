# Makefile for Microsoft SpaceV-10M dataset (Standard Track / non-OOD).
# Source: NeurIPS 2021 big-ann-benchmarks
#   https://big-ann-benchmarks.com/neurips21.html
#
# Format: 100-d int8 (signed, -128..127), L2 distance.
#   base   = SpaceV web-search document embeddings
#   query  = 29,316 in-distribution document embeddings (same model as
#            the base — this is the Standard Track, NOT the OOD track)
#
# Files (all under https://comp21storage.z5.web.core.windows.net/comp21/spacev1b/):
#   * spacev1b_base.i8bin         — full 1B base (~100 GB). We byte-range
#                                   the first 10M (1,000,000,008 bytes).
#   * query.i8bin                  — 29,316 × 100-d in-distribution queries.
#   * msspacev-gt-10M              — official top-200 GT for the 10M slice
#                                   (29,316 queries × 200 neighbours, int32).
#
# i8bin layout: [uint32 nrows][uint32 ncols][int8 × nrows × ncols].
# After byte-ranging the base, the on-disk header still claims nrows=1B;
# the converter takes --limit so it only mmaps the rows that actually
# exist on disk (no in-place header rewrite needed).

DATASET_NAME := spacev-10m
NUM_VECTORS  := 10000000

BASE_URL := https://comp21storage.z5.web.core.windows.net/comp21/spacev1b
# Header (8 bytes) + 10M × 100 × int8 = 1,000,000,008 bytes; range is inclusive.
BASE_RANGE := 0-1000000007

# Downloaded source files
BASE_I8BIN  := spacev1b_base.10M.i8bin
QUERY_I8BIN := query.i8bin
GT_BIN      := msspacev-gt-10M

# Final output files (.fvecs/.ivecs format)
BASE_FILE   := spacev10m_base.fvecs
QUERY_FILE  := spacev10m_query.fvecs
TRUTH_FILE  := spacev10m_groundtruth.ivecs

DOWNLOAD_FILES := $(BASE_I8BIN) $(QUERY_I8BIN) $(GT_BIN)
FINAL_FILES    := $(BASE_FILE) $(QUERY_FILE) $(TRUTH_FILE)

all: setup
.PHONY: all

setup: $(FINAL_FILES)
	@echo "-> $(DATASET_NAME) dataset setup complete!"
	@echo "   base : $(BASE_FILE) ($(NUM_VECTORS) × 100d, int8 widened to float32)"
	@echo "   query: $(QUERY_FILE) (29,316 × 100d, in-distribution)"
	@echo "   gt   : $(TRUTH_FILE) (29,316 queries × top-200 nbrs)"
	@echo "   metric: L2 (Standard Track, non-OOD)"

# ── downloads ─────────────────────────────────────────────────────────────────
# Byte-range the first 10M of the 1B base (~1 GB) instead of pulling the
# full 100 GB file.
$(BASE_I8BIN):
	@echo "-> Downloading first 10M of SpaceV base via byte-range (~1 GB)..."
	curl --fail --progress-bar -r $(BASE_RANGE) \
	    -o $(BASE_I8BIN) $(BASE_URL)/spacev1b_base.i8bin

$(QUERY_I8BIN):
	@echo "-> Downloading SpaceV in-distribution queries (~3 MB)..."
	curl --fail --progress-bar -o $(QUERY_I8BIN) $(BASE_URL)/$(QUERY_I8BIN)

$(GT_BIN):
	@echo "-> Downloading SpaceV-10M ground truth (~22 MB)..."
	curl --fail --progress-bar -o $(GT_BIN) $(BASE_URL)/$(GT_BIN)

# ── conversions ───────────────────────────────────────────────────────────────
$(BASE_FILE): $(BASE_I8BIN)
	@echo "-> Converting first $(NUM_VECTORS) base vectors i8bin → fvecs..."
	python3 ../i8bin_to_fvecs.py $(BASE_I8BIN) $(BASE_FILE) --limit $(NUM_VECTORS)

$(QUERY_FILE): $(QUERY_I8BIN)
	@echo "-> Converting query vectors i8bin → fvecs..."
	python3 ../i8bin_to_fvecs.py $(QUERY_I8BIN) $(QUERY_FILE)

$(TRUTH_FILE): $(GT_BIN)
	@echo "-> Converting ground truth bin → ivecs..."
	python3 ../bin_to_ivecs.py $(GT_BIN) $(TRUTH_FILE)

fetch: $(DOWNLOAD_FILES)

clean:
	rm -f $(FINAL_FILES)

clean-all: clean
	rm -f $(DOWNLOAD_FILES)

.PHONY: all setup fetch clean clean-all
