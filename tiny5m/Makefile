# Makefile for Tiny-5M dataset
# 5M base vectors, 384 dimensions, Euclidean distance
# Source: CUHK GQR benchmark suite

DATASET_NAME   := tiny5m
ARCHIVE_FILE   := tiny5m.tar.gz
URL            := https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/tiny5m.tar.gz
EXTRACT_DIR    := tiny5m

BASE_FILE      := tiny5m_base.fvecs
RAW_BASE_FILE  := tiny5m_base.raw.fvecs
QUERY_FILE     := tiny5m_query.fvecs
TRUTH_FILE     := tiny5m_groundtruth.ivecs
FINAL_FILES    := $(BASE_FILE) $(QUERY_FILE) $(TRUTH_FILE)

all: setup
.PHONY: all

include ../common.mk

setup: $(FINAL_FILES)
	@echo "-> $(DATASET_NAME) dataset setup complete!"

$(RAW_BASE_FILE) $(QUERY_FILE) &: $(ARCHIVE_FILE)
	@echo "-> Extracting $(ARCHIVE_FILE)..."
	@set -eu; stage=$$(mktemp -d .extract.XXXXXX); \
	trap 'rm -rf "$$stage"' EXIT HUP INT TERM; \
	tar --warning=no-alone-zero-block -xzf $(ARCHIVE_FILE) -C "$$stage" \
		$(EXTRACT_DIR)/$(BASE_FILE) $(EXTRACT_DIR)/$(QUERY_FILE); \
	mv "$$stage/$(EXTRACT_DIR)/$(BASE_FILE)" $(RAW_BASE_FILE); \
	mv "$$stage/$(EXTRACT_DIR)/$(QUERY_FILE)" $(QUERY_FILE); \
	touch $(RAW_BASE_FILE) $(QUERY_FILE)

include ../deduplicate.mk

clean:
	rm -f $(FINAL_FILES)

.PHONY: all setup clean
