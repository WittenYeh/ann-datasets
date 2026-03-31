# Makefile for Word2Vec dataset
# 1M base vectors, 300 dimensions, Euclidean distance
# Source: CUHK GQR benchmark suite

DATASET_NAME   := word2vec
ARCHIVE_FILE   := word2vec.tar.gz
URL            := https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/word2vec.tar.gz
EXTRACT_DIR    := word2vec

BASE_FILE      := word2vec_base.fvecs
QUERY_FILE     := word2vec_query.fvecs
TRUTH_FILE     := word2vec_groundtruth.ivecs
FINAL_FILES    := $(BASE_FILE) $(QUERY_FILE) $(TRUTH_FILE)

all: setup
.PHONY: all

include ../common.mk

setup: $(FINAL_FILES)
	@echo "-> $(DATASET_NAME) dataset setup complete!"

$(BASE_FILE) $(QUERY_FILE): $(ARCHIVE_FILE)
	@echo "-> Extracting $(ARCHIVE_FILE)..."
	tar -xzf $(ARCHIVE_FILE)
	mv $(EXTRACT_DIR)/* .
	rm -rf $(EXTRACT_DIR)

$(TRUTH_FILE): $(BASE_FILE) $(QUERY_FILE)
	@echo "-> Computing ground truth..."
	python3 ../compute_groundtruth.py $(BASE_FILE) $(QUERY_FILE) $(TRUTH_FILE) --k 100

clean:
	rm -f $(FINAL_FILES)

.PHONY: all setup clean
