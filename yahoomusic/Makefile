# Makefile for Yahoo Music dataset
# ~1.8M vectors, 300 dimensions, Euclidean distance
# Source: CUHK GQR benchmark suite
# Note: Archive contains a single file (yahoomusic_query_all.fvecs) with all
#       vectors. We randomly extract 1000 as query and remove them from base.

DATASET_NAME   := yahoomusic
ARCHIVE_FILE   := yahoomusic.tar.gz
URL            := https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/yahoomusic.tar.gz
EXTRACT_DIR    := yahoomusic
NUM_QUERIES    := 1000

ALL_FILE       := yahoomusic_query_all.fvecs
BASE_FILE      := yahoomusic_base.fvecs
QUERY_FILE     := yahoomusic_query.fvecs
TRUTH_FILE     := yahoomusic_groundtruth.ivecs
FINAL_FILES    := $(BASE_FILE) $(QUERY_FILE) $(TRUTH_FILE)

all: setup
.PHONY: all

include ../common.mk

setup: $(FINAL_FILES)
	@echo "-> $(DATASET_NAME) dataset setup complete!"

$(ALL_FILE): $(ARCHIVE_FILE)
	@echo "-> Extracting $(ARCHIVE_FILE)..."
	tar --warning=no-alone-zero-block -xzf $(ARCHIVE_FILE)
	mv $(EXTRACT_DIR)/$(ALL_FILE) .
	rm -rf $(EXTRACT_DIR)

$(BASE_FILE) $(QUERY_FILE) &: $(ALL_FILE)
	@echo "-> Extracting $(NUM_QUERIES) random query vectors from base..."
	python3 ../extract_query.py $(ALL_FILE) $(BASE_FILE) $(QUERY_FILE) \
		--num-queries $(NUM_QUERIES) --seed 42

$(TRUTH_FILE): $(BASE_FILE) $(QUERY_FILE)
	@echo "-> Computing ground truth..."
	python3 ../compute_groundtruth.py $(BASE_FILE) $(QUERY_FILE) $(TRUTH_FILE) --k 100

clean:
	rm -f $(FINAL_FILES) $(ALL_FILE)

.PHONY: all setup clean
