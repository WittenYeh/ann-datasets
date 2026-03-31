# Makefile for GloVe dataset (GloVe Twitter 100d word vectors)
# ~1.2M base vectors, 100 dimensions, Euclidean distance
# Homepage: http://nlp.stanford.edu/projects/glove/
# Source: hosted at CUHK GQR

# --- 1. Dataset-Specific Configuration ---
DATASET_NAME   := glove
ARCHIVE_FILE   := glove1.2m.tar.gz
URL            := https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/glove1.2m.tar.gz
EXTRACT_DIR    := glove1.2m

# Expected files after extraction (inside glove1.2m/ subdirectory)
BASE_FILE      := glove1.2m_base.fvecs
QUERY_FILE     := glove1.2m_query.fvecs
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
		echo "-> Downloading $(DATASET_NAME) (~838MB)..."; \
		wget --progress=bar:force:noscroll $(URL) -O $(ARCHIVE_FILE); \
	fi

# Extract
$(FINAL_FILES): $(ARCHIVE_FILE)
	@if [ ! -f "$(firstword $(FINAL_FILES))" ]; then \
		echo "-> Extracting files from $(ARCHIVE_FILE)..."; \
		tar --warning=no-alone-zero-block -xzf $(ARCHIVE_FILE); \
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
