# Makefile for SIFT10K dataset (subset of SIFT-1M)

DATASET_NAME := sift-10k
NUM_VECTORS  := 10000

SIFT1M_DIR     := ../sift-1m
SOURCE_BASE    := $(SIFT1M_DIR)/sift_base.fvecs
SOURCE_QUERY   := $(SIFT1M_DIR)/sift_query.fvecs
SOURCE_LEARN   := $(SIFT1M_DIR)/sift_learn.fvecs

BASE_FILE      := sift10k_base.fvecs
QUERY_FILE     := sift10k_query.fvecs
LEARN_FILE     := sift10k_learn.fvecs
TRUTH_FILE     := sift10k_groundtruth.ivecs
FINAL_FILES    := $(BASE_FILE) $(QUERY_FILE) $(LEARN_FILE) $(TRUTH_FILE)

all: setup
.PHONY: all

$(SOURCE_BASE) $(SOURCE_QUERY) $(SOURCE_LEARN):
	@echo "-> SIFT-1M source not found, building sift-1m first..."
	$(MAKE) -C $(SIFT1M_DIR) setup

setup: $(FINAL_FILES)
	@echo "-> $(DATASET_NAME) dataset setup complete!"

$(BASE_FILE): $(SOURCE_BASE)
	@echo "-> Extracting $(NUM_VECTORS) base vectors..."
	python3 ../extract_subset.py $(SOURCE_BASE) $(BASE_FILE) $(NUM_VECTORS)

$(QUERY_FILE): $(SOURCE_QUERY)
	cp $(SOURCE_QUERY) $(QUERY_FILE)

$(LEARN_FILE): $(SOURCE_LEARN)
	python3 ../extract_subset.py $(SOURCE_LEARN) $(LEARN_FILE) $(NUM_VECTORS)

$(TRUTH_FILE): $(BASE_FILE) $(QUERY_FILE)
	@echo "-> Computing ground truth for $(DATASET_NAME)..."
	python3 ../compute_groundtruth.py $(BASE_FILE) $(QUERY_FILE) $(TRUTH_FILE) --k 100

clean:
	rm -f $(FINAL_FILES)

.PHONY: all setup clean
