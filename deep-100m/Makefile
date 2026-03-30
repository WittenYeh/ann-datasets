# Makefile for Deep100M dataset (subset of Deep1B)
# Requires deep-1b to be downloaded first; triggers it automatically if needed.

# --- 1. Dataset-Specific Configuration ---
DATASET_NAME := deep-100m
NUM_VECTORS  := 100000000

# Source files from deep-1b
DEEP1B_DIR     := ../deep-1b
SOURCE_BASE    := $(DEEP1B_DIR)/deep1B_base.fvecs
SOURCE_QUERY   := $(DEEP1B_DIR)/deep1B_queries.fvecs
SOURCE_LEARN   := $(DEEP1B_DIR)/deep1B_learn.fvecs
SOURCE_GT      := $(DEEP1B_DIR)/deep1B_groundtruth.ivecs

# Output files (100M subset)
BASE_FILE      := deep100m_base.fvecs
QUERY_FILE     := deep100m_query.fvecs
LEARN_FILE     := deep100m_learn.fvecs
TRUTH_FILE     := deep100m_groundtruth.ivecs

FINAL_FILES    := $(BASE_FILE) $(QUERY_FILE) $(LEARN_FILE) $(TRUTH_FILE)
# --- End of Configuration ---

all: setup
.PHONY: all

# Ensure deep-1b source data exists
$(SOURCE_BASE) $(SOURCE_QUERY) $(SOURCE_LEARN) $(SOURCE_GT):
	@echo "-> Deep1B source not found, building deep-1b first..."
	$(MAKE) -C $(DEEP1B_DIR) setup

setup: $(FINAL_FILES)
	@echo "-> Deep100M dataset setup complete!"

# Extract subset (100M vectors) from deep-1b
$(BASE_FILE): $(SOURCE_BASE)
	@echo "-> Extracting $(NUM_VECTORS) base vectors from Deep1B..."
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
