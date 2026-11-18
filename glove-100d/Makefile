# Makefile for GloVe-100d dataset (Wikipedia + Gigaword, 2024 edition)
# ~1.2M vectors, 100 dimensions, Euclidean distance
# Source: https://nlp.stanford.edu/projects/glove/

DATASET_NAME   := glove-100d
ARCHIVE_FILE   := glove.2024.wikigiga.100d.zip
URL            := https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.100d.zip
TXT_FILE       := glove100d.txt
NUM_QUERIES    := 10000

FULL_BASE_FILE := glove100d_base_orig.fvecs
BASE_FILE      := glove100d_base.fvecs
RAW_BASE_FILE  := glove100d_base.raw.fvecs
QUERY_FILE     := glove100d_query.fvecs
TRUTH_FILE     := glove100d_groundtruth.ivecs
FINAL_FILES    := $(BASE_FILE) $(QUERY_FILE) $(TRUTH_FILE)

all: setup
.PHONY: all

setup: $(FINAL_FILES)
	@echo "-> $(DATASET_NAME) dataset setup complete!"

# Download
$(ARCHIVE_FILE):
	@echo "-> Downloading $(DATASET_NAME) (~560MB)..."
	wget --progress=bar:force:noscroll $(URL) -O $(ARCHIVE_FILE)

# Extract txt from zip and rename to predictable name
$(TXT_FILE): $(ARCHIVE_FILE)
	@if [ ! -f "$(TXT_FILE)" ]; then \
		echo "-> Extracting $(ARCHIVE_FILE)..."; \
		unzip -o $(ARCHIVE_FILE); \
		mv *_100_*combined.txt $(TXT_FILE); \
	fi

# Convert GloVe txt to fvecs
$(FULL_BASE_FILE): $(TXT_FILE)
	@set -eu; tmp=$$(mktemp "$@.XXXXXX"); \
	trap 'rm -f "$$tmp"' EXIT HUP INT TERM; \
	$(PYTHON) ../scripts/glove_to_fvecs.py "$<" "$$tmp"; mv -f "$$tmp" "$@"

include ../scripts/generate-queries.mk

# Compute ground truth
include ../scripts/deduplicate.mk

# Download only
fetch: $(ARCHIVE_FILE)

clean:
	rm -f $(FINAL_FILES) $(FULL_BASE_FILE)

clean-source:
	rm -f $(TXT_FILE)

clean-all: clean clean-source
	rm -f $(ARCHIVE_FILE)

.PHONY: all setup fetch clean clean-source clean-all
