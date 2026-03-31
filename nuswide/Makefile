# Makefile for NUS-WIDE dataset
# ~269K base vectors, 500 dimensions, Euclidean distance
# Source: CUHK GQR benchmark suite
# Note: Archive includes groundtruth with k=20; we recompute with k=100.

DATASET_NAME   := nuswide
ARCHIVE_FILE   := nuswide.tar.gz
URL            := https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/nuswide.tar.gz
EXTRACT_DIR    := nuswide

BASE_FILE      := nuswide_base.fvecs
QUERY_FILE     := nuswide_query.fvecs
TRUTH_FILE     := nuswide_groundtruth.ivecs
FINAL_FILES    := $(BASE_FILE) $(QUERY_FILE) $(TRUTH_FILE)

all: setup
.PHONY: all

include ../common.mk

setup: $(FINAL_FILES)
	@echo "-> $(DATASET_NAME) dataset setup complete!"

$(BASE_FILE) $(QUERY_FILE): $(ARCHIVE_FILE)
	@echo "-> Extracting $(ARCHIVE_FILE)..."
	tar -xzf $(ARCHIVE_FILE)
	mv $(EXTRACT_DIR)/$(BASE_FILE) $(EXTRACT_DIR)/$(QUERY_FILE) .
	rm -rf $(EXTRACT_DIR)

$(TRUTH_FILE): $(BASE_FILE) $(QUERY_FILE)
	@echo "-> Computing ground truth (k=100)..."
	python3 ../compute_groundtruth.py $(BASE_FILE) $(QUERY_FILE) $(TRUTH_FILE) --k 100

clean:
	rm -f $(FINAL_FILES)

.PHONY: all setup clean
