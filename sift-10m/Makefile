# Makefile for SIFT10M dataset (subset of SIFT1B/big-ann)
# Requires big-ann to be downloaded first; triggers it automatically if needed.

# --- 1. Dataset-Specific Configuration ---
DATASET_NAME := sift-10m
NUM_VECTORS  := 10000000

# Source files from big-ann (SIFT1B)
SIFT1B_DIR     := ../big-ann
SOURCE_BASE    := $(SIFT1B_DIR)/bigann_base.bvecs
SOURCE_QUERY   := $(SIFT1B_DIR)/bigann_query.bvecs
SOURCE_LEARN   := $(SIFT1B_DIR)/bigann_learn.bvecs
SOURCE_GT      := $(SIFT1B_DIR)/idx_1000M.ivecs

# Output files (10M subset)
BASE_FILE      := sift10m_base.bvecs
QUERY_FILE     := sift10m_query.bvecs
LEARN_FILE     := sift10m_learn.bvecs
TRUTH_FILE     := sift10m_groundtruth.ivecs

FINAL_FILES    := $(BASE_FILE) $(QUERY_FILE) $(LEARN_FILE) $(TRUTH_FILE)
# --- End of Configuration ---

all: setup
.PHONY: all

# Ensure SIFT1B source data exists
$(SOURCE_BASE) $(SOURCE_QUERY) $(SOURCE_LEARN) $(SOURCE_GT):
	@echo "-> SIFT1B source not found, building big-ann first..."
	$(MAKE) -C $(SIFT1B_DIR) setup

setup: $(FINAL_FILES)
	@echo "-> SIFT10M dataset setup complete!"

$(BASE_FILE): $(SOURCE_BASE)
	@echo "-> Extracting $(NUM_VECTORS) base vectors from SIFT1B..."
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
