# Makefile for CIFAR-60K dataset
# 60K base vectors, 512 dimensions, Euclidean distance
# Source: CUHK GQR benchmark suite

DATASET_NAME   := cifar60k
ARCHIVE_FILE   := cifar60k.tar.gz
URL            := https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/cifar60k.tar.gz
EXTRACT_DIR    := cifar60k

BASE_FILE      := cifar60k_base.fvecs
QUERY_FILE     := cifar60k_query.fvecs
TRUTH_FILE     := cifar60k_groundtruth.ivecs
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
