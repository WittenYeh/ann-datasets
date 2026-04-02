# Makefile for SIFT10M dataset (subset of SIFT1B/big-ann)

DATASET_NAME := sift-10m
NUM_VECTORS  := 10000000

SIFT1B_DIR     := ../big-ann
SOURCE_BASE    := $(SIFT1B_DIR)/bigann_base.bvecs
SOURCE_QUERY   := $(SIFT1B_DIR)/bigann_query.bvecs
SOURCE_LEARN   := $(SIFT1B_DIR)/bigann_learn.bvecs

BASE_FILE      := sift10m_base.bvecs
QUERY_FILE     := sift10m_query.bvecs
LEARN_FILE     := sift10m_learn.bvecs
TRUTH_FILE     := sift10m_groundtruth.ivecs
FINAL_FILES    := $(BASE_FILE) $(QUERY_FILE) $(LEARN_FILE) $(TRUTH_FILE)

all: setup
.PHONY: all

$(SOURCE_BASE) $(SOURCE_QUERY) $(SOURCE_LEARN):
	@echo "-> SIFT1B source not found, building big-ann first..."
	$(MAKE) -C $(SIFT1B_DIR) setup

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
