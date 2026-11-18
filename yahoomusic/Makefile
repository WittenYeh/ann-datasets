# Makefile for Yahoo Music dataset
# ~1.8M vectors, 300 dimensions, Euclidean distance
# Source: CUHK GQR benchmark suite
# The archive contains all base vectors in one file, yahoomusic_query_all.fvecs.

DATASET_NAME   := yahoomusic
ARCHIVE_FILE   := yahoomusic.tar.gz
URL            := https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/yahoomusic.tar.gz
EXTRACT_DIR    := yahoomusic
NUM_QUERIES    := 1000

FULL_BASE_FILE := yahoomusic_query_all.fvecs
BASE_FILE      := yahoomusic_base.fvecs
RAW_BASE_FILE  := yahoomusic_base.raw.fvecs
QUERY_FILE     := yahoomusic_query.fvecs
TRUTH_FILE     := yahoomusic_groundtruth.ivecs
FINAL_FILES    := $(BASE_FILE) $(QUERY_FILE) $(TRUTH_FILE)

all: setup
.PHONY: all

include ../scripts/common.mk

setup: $(FINAL_FILES)
	@echo "-> $(DATASET_NAME) dataset setup complete!"

$(FULL_BASE_FILE): $(ARCHIVE_FILE)
	@echo "-> Extracting $(ARCHIVE_FILE)..."
	@set -eu; stage=$$(mktemp -d .extract.XXXXXX); \
	trap 'rm -rf "$$stage"' EXIT HUP INT TERM; \
	tar --warning=no-alone-zero-block -xzf $(ARCHIVE_FILE) -C "$$stage" $(EXTRACT_DIR)/$(FULL_BASE_FILE); \
	mv "$$stage/$(EXTRACT_DIR)/$(FULL_BASE_FILE)" $(FULL_BASE_FILE); \
	touch $(FULL_BASE_FILE)

include ../scripts/generate-queries.mk
include ../scripts/deduplicate.mk

clean:
	rm -f $(FINAL_FILES) $(FULL_BASE_FILE)

.PHONY: all setup clean
