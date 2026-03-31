# Makefile for Tiny-5M dataset
# 5M base vectors, 384 dimensions, Euclidean distance
# Source: CUHK GQR benchmark suite

DATASET_NAME   := tiny5m
ARCHIVE_FILE   := tiny5m.tar.gz
URL            := https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/tiny5m.tar.gz
EXTRACT_DIR    := tiny5m

BASE_FILE      := tiny5m_base.fvecs
QUERY_FILE     := tiny5m_query.fvecs
TRUTH_FILE     := tiny5m_groundtruth.ivecs
FINAL_FILES    := $(BASE_FILE) $(QUERY_FILE) $(TRUTH_FILE)

all: setup
.PHONY: all

include ../common.mk

setup: $(FINAL_FILES)
	@echo "-> $(DATASET_NAME) dataset setup complete!"

$(BASE_FILE) $(QUERY_FILE) &: $(ARCHIVE_FILE)
	@echo "-> Extracting $(ARCHIVE_FILE)..."
	tar --warning=no-alone-zero-block -xzf $(ARCHIVE_FILE)
	mv $(EXTRACT_DIR)/* .
	rm -rf $(EXTRACT_DIR)

$(TRUTH_FILE): $(BASE_FILE) $(QUERY_FILE)
	@echo "-> Computing ground truth (5M vectors, this may take a while)..."
	python3 ../compute_groundtruth.py $(BASE_FILE) $(QUERY_FILE) $(TRUTH_FILE) --k 100

clean:
	rm -f $(FINAL_FILES)

.PHONY: all setup clean
