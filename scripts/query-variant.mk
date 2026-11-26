# Opt-in generated queries for a dataset with published queries. Keep its
# original files usable, including when another dataset uses them as a parent.
ifneq ($(query-gen),drawout)
ifneq ($(query-gen),gaussian)
$(error Unsupported query-gen='$(query-gen)'. Choose published, drawout or gaussian)
endif
endif

PUBLISHED_RAW := $(RAW_BASE_FILE)
PUBLISHED_INPUT := $(BASE_FILE)
ifeq ($(query-gen),drawout)
PUBLISHED_INPUT := $(or $(PUBLISHED_RAW),$(BASE_FILE))
endif

VARIANT_DIR := variants/$(query-gen)
ifeq ($(origin num_queries),undefined)
COUNT_QUERY := $(QUERY_FILE)
COUNT_FILE := $(VARIANT_DIR)/published-query-count
num_queries = $(file <$(COUNT_FILE))

$(COUNT_FILE): $(COUNT_QUERY) | $(VARIANT_DIR)
	@set -eu; count=$$(PYTHONPATH=../scripts $(PYTHON) -c \
		'import sys; from vecs_io import mmap_by_ext; print(len(mmap_by_ext(sys.argv[1])))' "$<"); \
	tmp=$$(mktemp "$@.XXXXXX"); trap 'rm -f "$$tmp"' EXIT HUP INT TERM; \
	printf '%s\n' "$$count" > "$$tmp"; mv -f "$$tmp" "$@"
endif

# Recheck the publisher's own prerequisites. Some archives contain both files.
$(PUBLISHED_INPUT) $(COUNT_QUERY) &: FORCE_PUBLISHED_INPUTS
	+$(MAKE) --no-print-directory query-gen=published "$(PUBLISHED_INPUT)"
ifneq ($(COUNT_QUERY),)
	+$(MAKE) --no-print-directory query-gen=published "$(COUNT_QUERY)"
endif

BASE_FILE := $(VARIANT_DIR)/base$(suffix $(PUBLISHED_INPUT))
TRUTH_FILE := $(VARIANT_DIR)/groundtruth.ivecs
RAW_BASE_FILE :=
ifeq ($(query-gen),drawout)
FULL_BASE_FILE := $(PUBLISHED_INPUT)
QUERY_FILE := $(VARIANT_DIR)/query$(suffix $(PUBLISHED_INPUT))
ifneq ($(PUBLISHED_RAW),)
RAW_BASE_FILE := $(VARIANT_DIR)/base.raw$(suffix $(PUBLISHED_INPUT))
endif
else
QUERY_FILE := $(VARIANT_DIR)/query.fvecs
$(BASE_FILE): $(PUBLISHED_INPUT) | $(VARIANT_DIR)
	ln -srf "$<" "$@"
endif

GT_K ?= $(if $(PUBLISHED_RAW),1000,100)
include ../scripts/generate-queries.mk
include ../scripts/deduplicate.mk

$(VARIANT_DIR):
	mkdir -p "$@"
$(BASE_CONFIG) $(QUERY_CONFIG) $(GT_CONFIG): | $(VARIANT_DIR)
$(QUERY_CONFIG): | $(COUNT_FILE)

setup: groundtruth
	@echo "Prepared $(BASE_FILE), $(QUERY_FILE), $(TRUTH_FILE)"

fetch: $(PUBLISHED_INPUT) $(COUNT_QUERY)

info:
	@for file in "$(BASE_FILE)" "$(QUERY_FILE)" "$(TRUTH_FILE)"; do \
		if test -f "$$file"; then $(PYTHON) ../scripts/preview_dataset.py "$$file"; fi; \
	done

clean:
	rm -f "$(BASE_FILE)" "$(QUERY_FILE)" "$(TRUTH_FILE)" "$(VARIANT_DIR)/published-query-count"

.PHONY: all setup fetch info clean FORCE_PUBLISHED_INPUTS
