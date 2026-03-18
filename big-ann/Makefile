# Makefile for the SIFT 1 Billion dataset

# --- 1. Dataset-Specific Configuration ---
DATASET_NAME := sift-1b

# SIFT-1B consists of multiple files that need to be downloaded separately
BASE_FILE_GZ   := bigann_base.bvecs.gz
LEARN_FILE_GZ  := bigann_learn.bvecs.gz
QUERY_FILE_GZ  := bigann_query.bvecs.gz
GND_ARCHIVE    := bigann_gnd.tar.gz

BASE_URL := ftp://ftp.irisa.fr/local/texmex/corpus

# Decompressed file names
BASE_FILE      := bigann_base.bvecs
LEARN_FILE     := bigann_learn.bvecs
QUERY_FILE     := bigann_query.bvecs
TRUTH_FILE     := idx_1000M.ivecs

# All files to download
DOWNLOAD_FILES := $(BASE_FILE_GZ) $(LEARN_FILE_GZ) $(QUERY_FILE_GZ) $(GND_ARCHIVE)
# All final decompressed files
FINAL_FILES    := $(BASE_FILE) $(LEARN_FILE) $(QUERY_FILE) $(TRUTH_FILE)
# --- End of Configuration ---


# --- 2. Include Generic Recipes ---
include ../common.mk
# --- End of Recipes ---


# --- 3. Define Targets ---
all: setup

setup: $(FINAL_FILES)
	@echo "-> SIFT-1B dataset setup complete!"

# Download rules for each file
$(BASE_FILE_GZ):
	@echo "-> Downloading base vectors (WARNING: ~120GB file, this will take a long time)..."
	wget --progress=bar:force:noscroll $(BASE_URL)/$(BASE_FILE_GZ) -O $(BASE_FILE_GZ)

$(LEARN_FILE_GZ):
	@echo "-> Downloading learn vectors (~37GB)..."
	wget --progress=bar:force:noscroll $(BASE_URL)/$(LEARN_FILE_GZ) -O $(LEARN_FILE_GZ)

$(QUERY_FILE_GZ):
	@echo "-> Downloading query vectors (small file)..."
	wget --progress=bar:force:noscroll $(BASE_URL)/$(QUERY_FILE_GZ) -O $(QUERY_FILE_GZ)

$(GND_ARCHIVE):
	@echo "-> Downloading ground truth..."
	wget --progress=bar:force:noscroll $(BASE_URL)/$(GND_ARCHIVE) -O $(GND_ARCHIVE)

# Decompression rules
$(BASE_FILE): $(BASE_FILE_GZ)
	@if [ ! -f "$(BASE_FILE)" ]; then \
		echo "-> Decompressing base vectors (this will take time)..."; \
		gunzip -k $(BASE_FILE_GZ); \
	else \
		echo "-> $(BASE_FILE) already exists, skipping decompression."; \
	fi

$(LEARN_FILE): $(LEARN_FILE_GZ)
	@if [ ! -f "$(LEARN_FILE)" ]; then \
		echo "-> Decompressing learn vectors..."; \
		gunzip -k $(LEARN_FILE_GZ); \
	else \
		echo "-> $(LEARN_FILE) already exists, skipping decompression."; \
	fi

$(QUERY_FILE): $(QUERY_FILE_GZ)
	@if [ ! -f "$(QUERY_FILE)" ]; then \
		echo "-> Decompressing query vectors..."; \
		gunzip -k $(QUERY_FILE_GZ); \
	else \
		echo "-> $(QUERY_FILE) already exists, skipping decompression."; \
	fi

$(TRUTH_FILE): $(GND_ARCHIVE)
	@if [ ! -f "$(TRUTH_FILE)" ]; then \
		echo "-> Extracting ground truth from archive..."; \
		tar -xzf $(GND_ARCHIVE); \
	else \
		echo "-> $(TRUTH_FILE) already exists, skipping extraction."; \
	fi

# Download only (no decompression)
fetch: $(DOWNLOAD_FILES)

# Inspect dataset information
info:
	@echo "=== SIFT-1B Dataset Information ==="
	@if [ -f "$(BASE_FILE)" ]; then \
		python3 ../preview_dataset.py $(BASE_FILE); \
	else \
		echo "Base file not found. Run 'make setup' first."; \
	fi
	@if [ -f "$(QUERY_FILE)" ]; then \
		python3 ../preview_dataset.py $(QUERY_FILE); \
	else \
		echo "Query file not found. Run 'make setup' first."; \
	fi
	@if [ -f "$(TRUTH_FILE)" ]; then \
		python3 ../preview_dataset.py $(TRUTH_FILE); \
	else \
		echo "Ground truth file not found. Run 'make setup' first."; \
	fi

# Clean decompressed files
clean:
	@echo "-> Removing decompressed data files..."
	rm -f $(FINAL_FILES)

# Clean everything including downloads
clean-all: clean
	@echo "-> Removing downloaded archives..."
	rm -f $(DOWNLOAD_FILES)

.PHONY: all setup fetch info clean clean-all