# Makefile for SIFT 100M dataset (subset of SIFT1B)

# --- 1. Dataset-Specific Configuration ---
DATASET_NAME := sift-100m
NUM_VECTORS  := 100000000

# Source files from SIFT1B
SIFT1B_DIR   := ../sift-1b
SOURCE_BASE  := $(SIFT1B_DIR)/bigann_base.bvecs
SOURCE_QUERY := $(SIFT1B_DIR)/bigann_query.bvecs
SOURCE_LEARN := $(SIFT1B_DIR)/bigann_learn.bvecs

# Output files
BASE_FILE    := sift100m_base.bvecs
QUERY_FILE   := sift100m_query.bvecs
LEARN_FILE   := sift100m_learn.bvecs

FINAL_FILES  := $(BASE_FILE) $(QUERY_FILE) $(LEARN_FILE)
# --- End of Configuration ---

all: setup

setup: $(FINAL_FILES)
	@echo "-> SIFT100M dataset setup complete!"

# Extract base vectors (100M from 1B)
$(BASE_FILE): $(SOURCE_BASE)
	@if [ ! -f "$(BASE_FILE)" ]; then \
		echo "-> Extracting $(NUM_VECTORS) base vectors from SIFT1B..."; \
		echo "   This will take a while (~13GB output)..."; \
		python3 ../extract_subset.py $(SOURCE_BASE) $(BASE_FILE) $(NUM_VECTORS); \
	else \
		echo "-> $(BASE_FILE) already exists, skipping extraction."; \
	fi

# Copy query vectors (same as SIFT1B)
$(QUERY_FILE): $(SOURCE_QUERY)
	@if [ ! -f "$(QUERY_FILE)" ]; then \
		echo "-> Copying query vectors from SIFT1B..."; \
		cp $(SOURCE_QUERY) $(QUERY_FILE); \
	else \
		echo "-> $(QUERY_FILE) already exists, skipping copy."; \
	fi

# Extract learn vectors (100M from SIFT1B learn set)
$(LEARN_FILE): $(SOURCE_LEARN)
	@if [ ! -f "$(LEARN_FILE)" ]; then \
		echo "-> Extracting learn vectors from SIFT1B..."; \
		python3 ../extract_subset.py $(SOURCE_LEARN) $(LEARN_FILE) $(NUM_VECTORS); \
	else \
		echo "-> $(LEARN_FILE) already exists, skipping extraction."; \
	fi

# Auto-initialize SIFT1B if source files don't exist
$(SOURCE_BASE) $(SOURCE_QUERY) $(SOURCE_LEARN):
	@echo "-> SIFT1B source files not found. Initializing SIFT1B..."
	@$(MAKE) -C $(SIFT1B_DIR) setup

# Inspect dataset information
info:
	@echo "=== $(DATASET_NAME) Dataset Information ==="
	@for file in $(FINAL_FILES); do \
		if [ -f "$$file" ]; then \
			python3 ../preview_dataset.py $$file; \
		fi; \
	done

clean:
	@echo "-> Removing extracted data files..."
	rm -f $(FINAL_FILES)

.PHONY: all setup info clean
