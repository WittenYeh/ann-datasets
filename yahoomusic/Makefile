# Makefile for Yahoo Music dataset
# ~1.8M vectors, 300 dimensions, Euclidean distance
# Source: CUHK GQR benchmark suite
# Note: Archive contains a single file (yahoomusic_query_all.fvecs) with all
#       vectors. We split the last 1000 as query and the rest as base.

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
	tar -xzf $(ARCHIVE_FILE)
	mv $(EXTRACT_DIR)/$(ALL_FILE) .
	rm -rf $(EXTRACT_DIR)

$(BASE_FILE) $(QUERY_FILE): $(ALL_FILE)
	@echo "-> Splitting into base and query (last $(NUM_QUERIES) as query)..."
	python3 -c "\
import sys; sys.path.insert(0, '..');\
from vecs_io import fvecs_mmap, fvecs_write;\
import numpy as np;\
data = fvecs_mmap('$(ALL_FILE)');\
n = data.shape[0];\
nq = $(NUM_QUERIES);\
fvecs_write('$(BASE_FILE)', np.array(data[:n-nq]));\
fvecs_write('$(QUERY_FILE)', np.array(data[n-nq:]));\
print(f'  Base: {n-nq} vectors, Query: {nq} vectors')"

$(TRUTH_FILE): $(BASE_FILE) $(QUERY_FILE)
	@echo "-> Computing ground truth..."
	python3 ../compute_groundtruth.py $(BASE_FILE) $(QUERY_FILE) $(TRUTH_FILE) --k 100

clean:
	rm -f $(FINAL_FILES) $(ALL_FILE)

.PHONY: all setup clean
