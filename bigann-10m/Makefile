# Makefile for SIFT10M dataset (subset of SIFT1B/bigann), float32 build.
#
# Pipeline per file: TRUNCATE the first 10M vectors out of the 1B SIFT
# base/learn (uint8 .bvecs), THEN UPCAST uint8 -> float32 (.fvecs). The
# float build is loadable by readers that only accept .fvecs/.ivecs (e.g.
# the artea VectorArray, which cannot read .bvecs); the FAISS-backed
# baselines read it via their .fvecs path.
#
# The uint8 -> float32 widening is LOSSLESS for SIFT: values 0..255 are
# exact in float32 and squared-L2 over 128 dims stays < 2^24, so neighbour
# rankings (and the ground truth ids) are identical to the integer metric.
# The .bvecs files are kept as cheap truncation intermediates (run
# `make clean-all` to drop them too).

DATASET_NAME := sift-10m
NUM_VECTORS  := 10000000

SIFT1B_DIR     := ../bigann
SOURCE_BASE    := $(SIFT1B_DIR)/bigann_base.bvecs
SOURCE_QUERY   := $(SIFT1B_DIR)/bigann_query.bvecs
SOURCE_LEARN   := $(SIFT1B_DIR)/bigann_learn.bvecs

# Truncated uint8 intermediates (first 10M).
BASE_BVECS     := sift10m_base.bvecs
QUERY_BVECS    := sift10m_query.bvecs
LEARN_BVECS    := sift10m_learn.bvecs

# Final float32 outputs (uint8 widened to float32).
BASE_FILE      := sift10m_base.fvecs
QUERY_FILE     := sift10m_query.fvecs
LEARN_FILE     := sift10m_learn.fvecs
TRUTH_FILE     := sift10m_groundtruth.ivecs

INTERMEDIATES  := $(BASE_BVECS) $(QUERY_BVECS) $(LEARN_BVECS)
FINAL_FILES    := $(BASE_FILE) $(QUERY_FILE) $(LEARN_FILE) $(TRUTH_FILE)

all: setup
.PHONY: all

$(SOURCE_BASE) $(SOURCE_QUERY) $(SOURCE_LEARN):
	@echo "-> SIFT1B source not found, building bigann first..."
	$(MAKE) -C $(SIFT1B_DIR) setup

setup: $(FINAL_FILES)
	@echo "-> $(DATASET_NAME) dataset setup complete!"
	@echo "   base : $(BASE_FILE) ($(NUM_VECTORS) x 128d, uint8 widened to float32)"
	@echo "   query: $(QUERY_FILE) (10,000 x 128d)"
	@echo "   gt   : $(TRUTH_FILE) (top-100 neighbours, int32)"
	@echo "   metric: L2"

# -- truncate (uint8 .bvecs intermediates) ----------------------------------
$(BASE_BVECS): $(SOURCE_BASE)
	@echo "-> Extracting first $(NUM_VECTORS) base vectors (uint8)..."
	python3 ../extract_subset.py $(SOURCE_BASE) $(BASE_BVECS) $(NUM_VECTORS)

$(QUERY_BVECS): $(SOURCE_QUERY)
	cp $(SOURCE_QUERY) $(QUERY_BVECS)

$(LEARN_BVECS): $(SOURCE_LEARN)
	@echo "-> Extracting first $(NUM_VECTORS) learn vectors (uint8)..."
	python3 ../extract_subset.py $(SOURCE_LEARN) $(LEARN_BVECS) $(NUM_VECTORS)

# -- upcast (uint8 -> float32 .fvecs) ----------------------------------------
$(BASE_FILE): $(BASE_BVECS)
	@echo "-> Upcasting base bvecs -> fvecs (uint8 -> float32)..."
	python3 ../bvecs_to_fvecs.py $(BASE_BVECS) $(BASE_FILE)

$(QUERY_FILE): $(QUERY_BVECS)
	@echo "-> Upcasting query bvecs -> fvecs..."
	python3 ../bvecs_to_fvecs.py $(QUERY_BVECS) $(QUERY_FILE)

$(LEARN_FILE): $(LEARN_BVECS)
	@echo "-> Upcasting learn bvecs -> fvecs..."
	python3 ../bvecs_to_fvecs.py $(LEARN_BVECS) $(LEARN_FILE)

# -- ground truth (computed on the float32 base/query; ids match uint8) -----
$(TRUTH_FILE): $(BASE_FILE) $(QUERY_FILE)
	@echo "-> Computing ground truth for $(DATASET_NAME)..."
	python3 ../compute_groundtruth.py $(BASE_FILE) $(QUERY_FILE) $(TRUTH_FILE) --k 100

fetch: $(INTERMEDIATES)

clean:
	rm -f $(FINAL_FILES)

clean-all: clean
	rm -f $(INTERMEDIATES)

.PHONY: all setup fetch clean clean-all
