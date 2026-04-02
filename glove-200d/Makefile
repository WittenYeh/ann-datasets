# Makefile for GloVe-200d dataset (Wikipedia + Gigaword, 2024 edition)
# ~1.2M vectors, 200 dimensions, Euclidean distance
# Source: https://nlp.stanford.edu/projects/glove/

DATASET_NAME   := glove-200d
ARCHIVE_FILE   := glove.2024.wikigiga.200d.zip
URL            := https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.200d.zip
TXT_FILE       := glove200d.txt
NUM_QUERIES    := 1000

ORIG_BASE_FILE := glove200d_base_orig.fvecs
BASE_FILE      := glove200d_base.fvecs
QUERY_FILE     := glove200d_query.fvecs
TRUTH_FILE     := glove200d_groundtruth.ivecs
FINAL_FILES    := $(BASE_FILE) $(QUERY_FILE) $(TRUTH_FILE)

all: setup
.PHONY: all

include ../common.mk

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
$(ORIG_BASE_FILE): $(TXT_FILE)
	@if [ ! -f "$(ORIG_BASE_FILE)" ]; then \
		echo "-> Converting GloVe txt to fvecs..."; \
		python3 ../glove_to_fvecs.py $(TXT_FILE) $(ORIG_BASE_FILE); \
	else \
		echo "-> $(ORIG_BASE_FILE) already exists, skipping conversion."; \
	fi

# Extract query vectors and compact base
$(BASE_FILE) $(QUERY_FILE) &: $(ORIG_BASE_FILE)
	@echo "-> Extracting $(NUM_QUERIES) random query vectors from base..."
	python3 ../extract_query.py $(ORIG_BASE_FILE) $(BASE_FILE) $(QUERY_FILE) \
		--num-queries $(NUM_QUERIES) --seed 42

# Compute ground truth
$(TRUTH_FILE): $(BASE_FILE) $(QUERY_FILE)
	@echo "-> Computing ground truth (k=100)..."
	python3 ../compute_groundtruth.py $(BASE_FILE) $(QUERY_FILE) $(TRUTH_FILE) --k 100

# Download only
fetch: $(ARCHIVE_FILE)

clean:
	rm -f $(FINAL_FILES) $(ORIG_BASE_FILE)

clean-source:
	rm -f $(TXT_FILE)

clean-all: clean clean-source
	rm -f $(ARCHIVE_FILE)

.PHONY: all setup fetch clean clean-source clean-all
