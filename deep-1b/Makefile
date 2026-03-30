# Makefile for Deep1B dataset (full 1 billion vectors)
# Source: BigANN Benchmarks (Yandex Research)
# Format: .fbin (downloaded) -> .fvecs (converted)

DATASET_NAME := deep-1b
BASE_URL := https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP

# Downloaded files (.fbin format)
BASE_FBIN      := base.1B.fbin
QUERY_FBIN     := query.public.10K.fbin
LEARN_FBIN     := learn.350M.fbin
GT_BIN         := ../deep_new_groundtruth.public.10K.bin
GT_BIN_URL     := https://storage.yandexcloud.net/yandex-research/ann-datasets/deep_new_groundtruth.public.10K.bin

# Final output files (.fvecs/.ivecs format)
BASE_FILE      := deep1B_base.fvecs
QUERY_FILE     := deep1B_queries.fvecs
LEARN_FILE     := deep1B_learn.fvecs
TRUTH_FILE     := deep1B_groundtruth.ivecs

DOWNLOAD_FILES := $(BASE_FBIN) $(QUERY_FBIN) $(LEARN_FBIN) $(GT_BIN)
FINAL_FILES    := $(BASE_FILE) $(QUERY_FILE) $(LEARN_FILE) $(TRUTH_FILE)

all: setup
.PHONY: all

setup: $(FINAL_FILES)
	@echo "-> Deep1B dataset setup complete!"

# Download rules
$(BASE_FBIN):
	@echo "-> Downloading Deep1B base vectors (WARNING: ~384GB file)..."
	wget --progress=bar:force:noscroll $(BASE_URL)/$(BASE_FBIN) -O $(BASE_FBIN)

$(QUERY_FBIN):
	@echo "-> Downloading Deep1B query vectors..."
	wget --progress=bar:force:noscroll $(BASE_URL)/$(QUERY_FBIN) -O $(QUERY_FBIN)

$(LEARN_FBIN):
	@echo "-> Downloading Deep1B learn vectors..."
	wget --progress=bar:force:noscroll $(BASE_URL)/$(LEARN_FBIN) -O $(LEARN_FBIN)

$(GT_BIN):
	@echo "-> Downloading Deep1B ground truth..."
	wget --progress=bar:force:noscroll $(GT_BIN_URL) -O $(GT_BIN)

# Convert .fbin -> .fvecs
$(BASE_FILE): $(BASE_FBIN)
	@echo "-> Converting base vectors to fvecs format..."
	python3 ../fbin_to_fvecs.py $(BASE_FBIN) $(BASE_FILE)

$(QUERY_FILE): $(QUERY_FBIN)
	@echo "-> Converting query vectors to fvecs format..."
	python3 ../fbin_to_fvecs.py $(QUERY_FBIN) $(QUERY_FILE)

$(LEARN_FILE): $(LEARN_FBIN)
	@echo "-> Converting learn vectors to fvecs format..."
	python3 ../fbin_to_fvecs.py $(LEARN_FBIN) $(LEARN_FILE)

# Convert .bin -> .ivecs
$(TRUTH_FILE): $(GT_BIN)
	@echo "-> Converting ground truth to ivecs format..."
	python3 ../bin_to_ivecs.py $(GT_BIN) $(TRUTH_FILE)

fetch: $(DOWNLOAD_FILES)

clean:
	rm -f $(FINAL_FILES)

clean-all: clean
	rm -f $(DOWNLOAD_FILES)

.PHONY: all setup fetch clean clean-all
