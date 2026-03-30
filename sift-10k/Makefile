# Makefile for SIFT10K dataset (subset of SIFT-1M)
# Requires sift-1m to be downloaded first; triggers it automatically if needed.

# --- 1. Dataset-Specific Configuration ---
DATASET_NAME := sift-10k
NUM_VECTORS  := 10000

# Source files from sift-1m
SIFT1M_DIR     := ../sift-1m
SOURCE_BASE    := $(SIFT1M_DIR)/sift_base.fvecs
SOURCE_QUERY   := $(SIFT1M_DIR)/sift_query.fvecs
SOURCE_LEARN   := $(SIFT1M_DIR)/sift_learn.fvecs
SOURCE_GT      := $(SIFT1M_DIR)/sift_groundtruth.ivecs

# Output files (10K subset)
BASE_FILE      := sift10k_base.fvecs
QUERY_FILE     := sift10k_query.fvecs
LEARN_FILE     := sift10k_learn.fvecs
TRUTH_FILE     := sift10k_groundtruth.ivecs

FINAL_FILES    := $(BASE_FILE) $(QUERY_FILE) $(LEARN_FILE) $(TRUTH_FILE)
# --- End of Configuration ---

all: setup
.PHONY: all

# Ensure sift-1m source data exists
$(SOURCE_BASE) $(SOURCE_QUERY) $(SOURCE_LEARN) $(SOURCE_GT):
	@echo "-> SIFT-1M source not found, building sift-1m first..."
	$(MAKE) -C $(SIFT1M_DIR) setup

setup: $(FINAL_FILES)
	@echo "-> SIFT10K dataset setup complete!"

$(BASE_FILE): $(SOURCE_BASE)
	@echo "-> Extracting $(NUM_VECTORS) base vectors from SIFT-1M..."
	python3 ../extract_subset.py $(SOURCE_BASE) $(BASE_FILE) $(NUM_VECTORS)

$(QUERY_FILE): $(SOURCE_QUERY)
	@echo "-> Copying query vectors..."
	cp $(SOURCE_QUERY) $(QUERY_FILE)

$(LEARN_FILE): $(SOURCE_LEARN)
	@echo "-> Extracting $(NUM_VECTORS) learn vectors..."
	python3 ../extract_subset.py $(SOURCE_LEARN) $(LEARN_FILE) $(NUM_VECTORS)

$(TRUTH_FILE): $(SOURCE_GT)
	@echo "-> Copying ground truth..."
	cp $(SOURCE_GT) $(TRUTH_FILE)

info:
	@echo "=== $(DATASET_NAME) Dataset Information ==="
	@for file in $(FINAL_FILES); do \
		if [ -f "$$file" ]; then python3 ../preview_dataset.py $$file; fi; \
	done

clean:
	@echo "-> Removing extracted data files..."
	rm -f $(FINAL_FILES)

.PHONY: all setup info clean
