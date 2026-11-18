# Makefile for GloVe-200d dataset (Wikipedia + Gigaword, 2024 edition)
# ~1.2M vectors, 200 dimensions, Euclidean distance
# Source: https://nlp.stanford.edu/projects/glove/

DATASET_NAME   := glove-200d
ARCHIVE_FILE   := glove.2024.wikigiga.200d.zip
URL            := https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.200d.zip
TXT_FILE       := glove200d.txt
NUM_QUERIES    := 1000

FULL_BASE_FILE := glove200d_base_orig.fvecs
BASE_FILE      := glove200d_base.fvecs
QUERY_FILE     := glove200d_query.fvecs
TRUTH_FILE     := glove200d_groundtruth.ivecs
FINAL_FILES    := $(BASE_FILE) $(QUERY_FILE) $(TRUTH_FILE)

all: setup
.PHONY: all

setup: $(FINAL_FILES)
	@echo "-> $(DATASET_NAME) dataset setup complete!"

# Download
$(ARCHIVE_FILE):
	@echo "-> Downloading $(DATASET_NAME) (~1.1GB)..."
	wget --progress=bar:force:noscroll $(URL) -O $(ARCHIVE_FILE)

# Extract txt from zip and rename to predictable name
$(TXT_FILE): $(ARCHIVE_FILE)
	@if [ ! -f "$(TXT_FILE)" ]; then \
		echo "-> Extracting $(ARCHIVE_FILE)..."; \
		unzip -o $(ARCHIVE_FILE); \
		mv *_200_*combined.txt $(TXT_FILE); \
	fi

# Convert GloVe txt to fvecs
$(FULL_BASE_FILE): $(TXT_FILE)
	@set -eu; tmp=$$(mktemp "$@.XXXXXX"); \
	trap 'rm -f "$$tmp"' EXIT HUP INT TERM; \
	$(PYTHON) ../scripts/glove_to_fvecs.py "$<" "$$tmp"; mv -f "$$tmp" "$@"

include ../scripts/generate-queries.mk

# Compute ground truth
$(TRUTH_FILE): $(BASE_FILE) $(QUERY_FILE) ../scripts/compute_groundtruth.py ../scripts/vecs_io.py
	@echo "-> Computing ground truth (k=100)..."
	OMP_NUM_THREADS=$$(nproc) numactl --interleave=all $(PYTHON) \
		../scripts/compute_groundtruth.py $(BASE_FILE) $(QUERY_FILE) $(TRUTH_FILE) --k 100

# Download only
fetch: $(ARCHIVE_FILE)

clean:
	rm -f $(FINAL_FILES) $(FULL_BASE_FILE)

clean-source:
	rm -f $(TXT_FILE)

clean-all: clean clean-source
	rm -f $(ARCHIVE_FILE)

.PHONY: all setup fetch clean clean-source clean-all
