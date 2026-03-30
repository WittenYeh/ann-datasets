# Makefile for Deep10M dataset (subset of Deep1B)

DATASET_NAME := deep-10m
NUM_VECTORS  := 10000000

DEEP1B_DIR     := ../deep-1b
SOURCE_BASE    := $(DEEP1B_DIR)/deep1B_base.fvecs
SOURCE_QUERY   := $(DEEP1B_DIR)/deep1B_queries.fvecs
SOURCE_LEARN   := $(DEEP1B_DIR)/deep1B_learn.fvecs

BASE_FILE      := deep10m_base.fvecs
QUERY_FILE     := deep10m_query.fvecs
LEARN_FILE     := deep10m_learn.fvecs
TRUTH_FILE     := deep10m_groundtruth.ivecs
FINAL_FILES    := $(BASE_FILE) $(QUERY_FILE) $(LEARN_FILE) $(TRUTH_FILE)

all: setup
.PHONY: all

$(SOURCE_BASE) $(SOURCE_QUERY) $(SOURCE_LEARN):
	@echo "-> Deep1B source not found, building deep-1b first..."
	$(MAKE) -C $(DEEP1B_DIR) setup

setup: $(FINAL_FILES)
	@echo "-> $(DATASET_NAME) dataset setup complete!"

$(BASE_FILE): $(SOURCE_BASE)
	@echo "-> Extracting $(NUM_VECTORS) base vectors..."
	python3 ../extract_subset.py $(SOURCE_BASE) $(BASE_FILE) $(NUM_VECTORS)

$(QUERY_FILE): $(SOURCE_QUERY)
	cp $(SOURCE_QUERY) $(QUERY_FILE)

$(LEARN_FILE): $(SOURCE_LEARN)
	python3 ../extract_subset.py $(SOURCE_LEARN) $(LEARN_FILE) $(NUM_VECTORS)

$(TRUTH_FILE): $(BASE_FILE) $(QUERY_FILE)
	@echo "-> Computing ground truth for $(DATASET_NAME)..."
	python3 ../compute_groundtruth.py $(BASE_FILE) $(QUERY_FILE) $(TRUTH_FILE) --k 100

clean:
	rm -f $(FINAL_FILES)

.PHONY: all setup clean
