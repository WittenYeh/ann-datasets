# Makefile for Deep1B dataset (full 1 billion vectors)

# --- 1. Dataset-Specific Configuration ---
DATASET_NAME := deep-1b

# Deep1B dataset URLs (from Yandex)
BASE_URL := https://storage.yandexcloud.net/yandex-research/ann-datasets/deep1b

# Downloaded files
BASE_FILE_GZ   := deep1B_base.fvecs.gz
QUERY_FILE_GZ  := deep1B_queries.fvecs.gz
LEARN_FILE_GZ  := deep1B_learn.fvecs.gz
GT_FILE_GZ     := deep1B_groundtruth.ivecs.gz

# Final decompressed files
BASE_FILE      := deep1B_base.fvecs
QUERY_FILE     := deep1B_queries.fvecs
LEARN_FILE     := deep1B_learn.fvecs
TRUTH_FILE     := deep1B_groundtruth.ivecs

DOWNLOAD_FILES := $(BASE_FILE_GZ) $(QUERY_FILE_GZ) $(LEARN_FILE_GZ) $(GT_FILE_GZ)
FINAL_FILES    := $(BASE_FILE) $(QUERY_FILE) $(LEARN_FILE) $(TRUTH_FILE)
# --- End of Configuration ---

# --- 2. Define Default Target (must be before includes) ---
all: setup
.PHONY: all

# --- 3. Targets ---
setup: $(FINAL_FILES)
	@echo "-> Deep1B dataset setup complete!"

# Download rules
$(BASE_FILE_GZ):
	@echo "-> Downloading Deep1B base vectors (WARNING: ~112GB file)..."
	wget --progress=bar:force:noscroll $(BASE_URL)/$(BASE_FILE_GZ) -O $(BASE_FILE_GZ)

$(QUERY_FILE_GZ):
	@echo "-> Downloading Deep1B query vectors..."
	wget --progress=bar:force:noscroll $(BASE_URL)/$(QUERY_FILE_GZ) -O $(QUERY_FILE_GZ)

$(LEARN_FILE_GZ):
	@echo "-> Downloading Deep1B learn vectors..."
	wget --progress=bar:force:noscroll $(BASE_URL)/$(LEARN_FILE_GZ) -O $(LEARN_FILE_GZ)

$(GT_FILE_GZ):
	@echo "-> Downloading Deep1B ground truth..."
	wget --progress=bar:force:noscroll $(BASE_URL)/$(GT_FILE_GZ) -O $(GT_FILE_GZ)

# Decompression rules
$(BASE_FILE): $(BASE_FILE_GZ)
	@if [ ! -f "$(BASE_FILE)" ]; then \
		echo "-> Decompressing base vectors (this will take a long time)..."; \
		gunzip -k $(BASE_FILE_GZ); \
	else \
		echo "-> $(BASE_FILE) already exists, skipping decompression."; \
	fi

$(QUERY_FILE): $(QUERY_FILE_GZ)
	@if [ ! -f "$(QUERY_FILE)" ]; then \
		echo "-> Decompressing query vectors..."; \
		gunzip -k $(QUERY_FILE_GZ); \
	else \
		echo "-> $(QUERY_FILE) already exists, skipping decompression."; \
	fi

$(LEARN_FILE): $(LEARN_FILE_GZ)
	@if [ ! -f "$(LEARN_FILE)" ]; then \
		echo "-> Decompressing learn vectors..."; \
		gunzip -k $(LEARN_FILE_GZ); \
	else \
		echo "-> $(LEARN_FILE) already exists, skipping decompression."; \
	fi

$(TRUTH_FILE): $(GT_FILE_GZ)
	@if [ ! -f "$(TRUTH_FILE)" ]; then \
		echo "-> Decompressing ground truth..."; \
		gunzip -k $(GT_FILE_GZ); \
	else \
		echo "-> $(TRUTH_FILE) already exists, skipping decompression."; \
	fi

# Download only (no decompression)
fetch: $(DOWNLOAD_FILES)

# Inspect dataset information
info:
	@echo "=== $(DATASET_NAME) Dataset Information ==="
	@for file in $(FINAL_FILES); do \
		if [ -f "$$file" ]; then \
			python3 ../preview_dataset.py $$file; \
		fi; \
	done

# Clean decompressed files
clean:
	@echo "-> Removing decompressed data files..."
	rm -f $(FINAL_FILES)

# Clean everything including downloads
clean-all: clean
	@echo "-> Removing downloaded archives..."
	rm -f $(DOWNLOAD_FILES)

.PHONY: all setup fetch info clean clean-all
