# Shared pipeline: original base + publisher query (or fixed-seed split),
# exact base deduplication, then our own top-k L2 ground truth.
# Include after defining RAW_BASE_FILE, BASE_FILE, QUERY_FILE, TRUTH_FILE.
PYTHON ?= python3
GT_K ?= 1000
GT_CHUNK_SIZE ?= 2000000
DATASET_THREADS := $(shell nproc)
DATASET_RUN = env OMP_NUM_THREADS=$(DATASET_THREADS) NUMBA_NUM_THREADS=$(DATASET_THREADS) POLARS_MAX_THREADS=$(DATASET_THREADS) OPENBLAS_NUM_THREADS=$(DATASET_THREADS) MKL_NUM_THREADS=$(DATASET_THREADS) numactl --interleave=all $(PYTHON)
DEDUP_REPORT := $(BASE_FILE).dedup.json
GT_CONFIG := $(TRUTH_FILE).config

$(BASE_FILE) $(DEDUP_REPORT) &: $(RAW_BASE_FILE) ../deduplicate.py ../deduplicate.mk
	$(DATASET_RUN) ../deduplicate.py $(RAW_BASE_FILE) $(BASE_FILE) \
		--threads $(DATASET_THREADS) --report $(DEDUP_REPORT)

# Recompute when options change, without making an unchanged build redo GT.
$(GT_CONFIG): FORCE_GT_CONFIG
	@set -eu; tmp=$$(mktemp "$(GT_CONFIG).XXXXXX"); \
	trap 'rm -f "$$tmp"' EXIT HUP INT TERM; \
	printf '%s\n' 'k=$(GT_K)' 'chunk_size=$(GT_CHUNK_SIZE)' > "$$tmp"; \
	if ! cmp -s "$$tmp" "$@"; then mv "$$tmp" "$@"; fi

$(TRUTH_FILE): $(BASE_FILE) $(DEDUP_REPORT) $(QUERY_FILE) $(GT_CONFIG) ../compute_groundtruth.py ../vecs_io.py ../deduplicate.mk
	$(DATASET_RUN) ../compute_groundtruth.py $(BASE_FILE) $(QUERY_FILE) $(TRUTH_FILE) \
		--k $(GT_K) --chunk-size $(GT_CHUNK_SIZE)

deduplicate: $(BASE_FILE) $(DEDUP_REPORT)
groundtruth: $(TRUTH_FILE)

clean: clean-dedup
clean-all: clean
clean-dedup:
	rm -f $(RAW_BASE_FILE) $(DEDUP_REPORT) $(GT_CONFIG)

.PHONY: deduplicate groundtruth clean-dedup FORCE_GT_CONFIG
