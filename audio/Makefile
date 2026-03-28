# Makefile for Audio dataset
# ~54K base vectors, 192 dimensions, Euclidean distance
# Homepage: https://www.cs.princeton.edu/cass/demos.htm
# Source: hosted at CUHK GQR

# --- 1. Dataset-Specific Configuration ---
DATASET_NAME   := audio
ARCHIVE_FILE   := audio.tar.gz
URL            := https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/audio.tar.gz
EXTRACT_DIR    := audio

# Expected files after extraction (inside audio/ subdirectory)
BASE_FILE      := audio_base.fvecs
QUERY_FILE     := audio_query.fvecs
FINAL_FILES    := $(BASE_FILE) $(QUERY_FILE)
# --- End of Configuration ---

all: setup

setup: $(FINAL_FILES)
	@echo "-> $(DATASET_NAME) dataset setup complete!"

# Download
$(ARCHIVE_FILE):
	@if [ -f "$(ARCHIVE_FILE)" ]; then \
		echo "-> $(ARCHIVE_FILE) already exists, skipping download."; \
	else \
		echo "-> Downloading $(DATASET_NAME) (~25MB)..."; \
		wget --progress=bar:force:noscroll $(URL) -O $(ARCHIVE_FILE); \
	fi

# Extract
$(FINAL_FILES): $(ARCHIVE_FILE)
	@if [ ! -f "$(firstword $(FINAL_FILES))" ]; then \
		echo "-> Extracting files from $(ARCHIVE_FILE)..."; \
		tar -xzf $(ARCHIVE_FILE); \
		echo "-> Moving files from $(EXTRACT_DIR)/ subdirectory..."; \
		for file in $(FINAL_FILES); do \
			if [ -f "$(EXTRACT_DIR)/$$file" ]; then \
				mv $(EXTRACT_DIR)/$$file .; \
			elif [ -f "$$file" ]; then \
				echo "  $$file already in place."; \
			else \
				echo "Warning: $$file not found in archive."; \
			fi; \
		done; \
		echo "-> Cleaning up temporary directory '$(EXTRACT_DIR)'..."; \
		rm -rf $(EXTRACT_DIR); \
	else \
		echo "-> Files already extracted, skipping."; \
	fi

# Download only
fetch: $(ARCHIVE_FILE)

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

# Clean everything
clean-all: clean
	@echo "-> Removing downloaded archive..."
	rm -f $(ARCHIVE_FILE)

.PHONY: all setup fetch info clean clean-all
