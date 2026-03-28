# Makefile for CRAWL dataset (fastText Common Crawl word vectors)
# ~2M base vectors, 300 dimensions, Euclidean distance
# Homepage: http://commoncrawl.org/
# Source: fastText (Meta), text format converted to fvecs

# --- 1. Dataset-Specific Configuration ---
DATASET_NAME   := crawl
ARCHIVE_FILE   := crawl-300d-2M.vec.zip
VEC_FILE       := crawl-300d-2M.vec
URL            := https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip

BASE_FILE      := crawl_base.fvecs
FINAL_FILES    := $(BASE_FILE)
# --- End of Configuration ---

all: setup

setup: $(BASE_FILE)
	@echo "-> $(DATASET_NAME) base vectors ready!"
	@echo "NOTE: Query vectors and ground truth must be generated separately."

# Download
$(ARCHIVE_FILE):
	@if [ -f "$(ARCHIVE_FILE)" ]; then \
		echo "-> $(ARCHIVE_FILE) already exists, skipping download."; \
	else \
		echo "-> Downloading $(DATASET_NAME) (~1.5GB)..."; \
		wget --progress=bar:force:noscroll $(URL) -O $(ARCHIVE_FILE); \
	fi

# Decompress
$(VEC_FILE): $(ARCHIVE_FILE)
	@if [ ! -f "$(VEC_FILE)" ]; then \
		echo "-> Extracting $(ARCHIVE_FILE)..."; \
		unzip -o $(ARCHIVE_FILE); \
	else \
		echo "-> $(VEC_FILE) already exists, skipping extraction."; \
	fi

# Convert to fvecs
$(BASE_FILE): $(VEC_FILE)
	@if [ ! -f "$(BASE_FILE)" ]; then \
		echo "-> Converting .vec to .fvecs format..."; \
		python3 ../vec_to_fvecs.py $(VEC_FILE) $(BASE_FILE); \
	else \
		echo "-> $(BASE_FILE) already exists, skipping conversion."; \
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

# Clean converted files
clean:
	@echo "-> Removing converted data files..."
	rm -f $(FINAL_FILES)

# Clean source files
clean-source:
	@echo "-> Removing decompressed source file..."
	rm -f $(VEC_FILE)

# Clean everything
clean-all: clean clean-source
	@echo "-> Removing downloaded archive..."
	rm -f $(ARCHIVE_FILE)

.PHONY: all setup fetch info clean clean-source clean-all
