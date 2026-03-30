# Makefile for Deep100M dataset (subset of Deep1B)

# --- 1. Dataset-Specific Configuration ---
DATASET_NAME := deep-100m
NUM_VECTORS  := 100000000

# Deep1B dataset URLs (from Yandex)
BASE_URL := https://storage.yandexcloud.net/yandex-research/ann-datasets/deep1b

# Downloaded files
BASE_FILE_GZ   := deep1B_base.fvecs.gz
QUERY_FILE_GZ  := deep1B_queries.fvecs.gz
LEARN_FILE_GZ  := deep1B_learn.fvecs.gz
GT_FILE_GZ     := deep1B_groundtruth.ivecs.gz

# Decompressed source files
SOURCE_BASE    := deep1B_base.fvecs
SOURCE_QUERY   := deep1B_queries.fvecs
SOURCE_LEARN   := deep1B_learn.fvecs
SOURCE_GT      := deep1B_groundtruth.ivecs

# Output files (100M subset)
BASE_FILE      := deep100m_base.fvecs
QUERY_FILE     := deep100m_query.fvecs
LEARN_FILE     := deep100m_learn.fvecs
TRUTH_FILE     := deep100m_groundtruth.ivecs

DOWNLOAD_FILES := $(BASE_FILE_GZ) $(QUERY_FILE_GZ) $(LEARN_FILE_GZ) $(GT_FILE_GZ)
SOURCE_FILES   := $(SOURCE_BASE) $(SOURCE_QUERY) $(SOURCE_LEARN) $(SOURCE_GT)
FINAL_FILES    := $(BASE_FILE) $(QUERY_FILE) $(LEARN_FILE) $(TRUTH_FILE)
# --- End of Configuration ---

all: setup
.PHONY: all

setup: $(FINAL_FILES)
	@echo "-> Deep100M dataset setup complete!"

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
$(SOURCE_BASE): $(BASE_FILE_GZ)
	@if [ ! -f "$(SOURCE_BASE)" ]; then \
		echo "-> Decompressing base vectors..."; \
		gunzip -k $(BASE_FILE_GZ); \
	fi

$(SOURCE_QUERY): $(QUERY_FILE_GZ)
	@if [ ! -f "$(SOURCE_QUERY)" ]; then \
		echo "-> Decompressing query vectors..."; \
		gunzip -k $(QUERY_FILE_GZ); \
	fi

$(SOURCE_LEARN): $(LEARN_FILE_GZ)
	@if [ ! -f "$(SOURCE_LEARN)" ]; then \
		echo "-> Decompressing learn vectors..."; \
		gunzip -k $(LEARN_FILE_GZ); \
	fi

$(SOURCE_GT): $(GT_FILE_GZ)
	@if [ ! -f "$(SOURCE_GT)" ]; then \
		echo "-> Decompressing ground truth..."; \
		gunzip -k $(GT_FILE_GZ); \
	fi

# Extract subset (100M vectors)
$(BASE_FILE): $(SOURCE_BASE)
	@if [ ! -f "$(BASE_FILE)" ]; then \
		echo "-> Extracting $(NUM_VECTORS) base vectors from Deep1B..."; \
		python3 ../extract_subset.py $(SOURCE_BASE) $(BASE_FILE) $(NUM_VECTORS); \
	else \
		echo "-> $(BASE_FILE) already exists, skipping extraction."; \
	fi

$(QUERY_FILE): $(SOURCE_QUERY)
	@if [ ! -f "$(QUERY_FILE)" ]; then \
		echo "-> Copying query vectors..."; \
		cp $(SOURCE_QUERY) $(QUERY_FILE); \
	else \
		echo "-> $(QUERY_FILE) already exists, skipping copy."; \
	fi

$(LEARN_FILE): $(SOURCE_LEARN)
	@if [ ! -f "$(LEARN_FILE)" ]; then \
		echo "-> Extracting learn vectors..."; \
		python3 ../extract_subset.py $(SOURCE_LEARN) $(LEARN_FILE) $(NUM_VECTORS); \
	else \
		echo "-> $(LEARN_FILE) already exists, skipping extraction."; \
	fi

$(TRUTH_FILE): $(SOURCE_GT)
	@if [ ! -f "$(TRUTH_FILE)" ]; then \
		echo "-> Copying ground truth..."; \
		cp $(SOURCE_GT) $(TRUTH_FILE); \
	else \
		echo "-> $(TRUTH_FILE) already exists, skipping copy."; \
	fi

# Download only
fetch: $(DOWNLOAD_FILES)

# Inspect dataset information
info:
	@echo "=== $(DATASET_NAME) Dataset Information ==="
	@for file in $(FINAL_FILES); do \
		if [ -f "$$file" ]; then \
			python3 ../preview_dataset.py $$file; \
		fi; \
	done

# Clean extracted files
clean:
	@echo "-> Removing extracted data files..."
	rm -f $(FINAL_FILES)

# Clean source files
clean-source:
	@echo "-> Removing decompressed source files..."
	rm -f $(SOURCE_FILES)

# Clean everything
clean-all: clean clean-source
	@echo "-> Removing downloaded archives..."
	rm -f $(DOWNLOAD_FILES)

.PHONY: all setup fetch info clean clean-source clean-all
