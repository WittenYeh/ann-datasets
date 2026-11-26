# Select how to generate queries from the complete source.
PYTHON ?= python3
query-gen ?= drawout
num_queries ?= $(NUM_QUERIES)
variance ?= 0.0001
export ANN_QUERY_COUNT = $(num_queries)
export ANN_QUERY_VARIANCE = $(variance)

SAMPLE_BASE := $(or $(RAW_BASE_FILE),$(BASE_FILE))
BASE_CONFIG := $(SAMPLE_BASE).config
QUERY_CONFIG := $(QUERY_FILE).config

# Only a change of method invalidates the full base in Gaussian mode.
$(BASE_CONFIG): private config_fields = 'query-gen=$(query-gen)'
$(QUERY_CONFIG): private config_fields = 'query-gen=$(query-gen)' "num_queries=$$ANN_QUERY_COUNT" $(noise_config)
$(BASE_CONFIG) $(QUERY_CONFIG): FORCE_QUERY_CONFIG
	@set -eu; tmp=$$(mktemp "$@.XXXXXX"); \
	trap 'rm -f "$$tmp"' EXIT HUP INT TERM; \
	printf '%s\n' $(config_fields) > "$$tmp"; \
	if ! cmp -s "$$tmp" "$@"; then mv -f "$$tmp" "$@"; fi

ifeq ($(query-gen),drawout)
# Draw before deduplicating, so both files reproduce the original held-out split.
$(SAMPLE_BASE) $(QUERY_FILE) &: $(FULL_BASE_FILE) $(BASE_CONFIG) $(QUERY_CONFIG) ../scripts/extract_query.py ../scripts/vecs_io.py
	@set -eu; stage=$$(mktemp -d "$(dir $(SAMPLE_BASE)).drawout.XXXXXX"); \
	trap 'rm -rf "$$stage"' EXIT HUP INT TERM; \
	$(PYTHON) ../scripts/extract_query.py "$<" "$$stage/base" "$$stage/query" --num-queries "$$ANN_QUERY_COUNT"; \
	mv -f "$$stage/base" "$(SAMPLE_BASE)"; mv -f "$$stage/query" "$(QUERY_FILE)"

else ifeq ($(query-gen),gaussian)
noise_config = "variance=$$ANN_QUERY_VARIANCE"
ifneq ($(FULL_BASE_FILE),)
$(SAMPLE_BASE): $(FULL_BASE_FILE) $(BASE_CONFIG)
	@set -eu; tmp=$$(mktemp "$@.XXXXXX"); \
	trap 'rm -f "$$tmp"' EXIT HUP INT TERM; \
	cp --reflink=auto "$<" "$$tmp"; mv -f "$$tmp" "$@"
endif

$(QUERY_FILE): $(BASE_FILE) $(QUERY_CONFIG) ../scripts/generate_queries.py ../scripts/vecs_io.py
	$(PYTHON) ../scripts/generate_queries.py "$<" "$@" \
		--num-queries "$$ANN_QUERY_COUNT" --variance "$$ANN_QUERY_VARIANCE"
else
$(error Unsupported query-gen='$(query-gen)' for this dataset. Choose drawout or gaussian)
endif

clean: clean-query-config
clean-query-config:
	rm -f $(QUERY_CONFIG) $(BASE_CONFIG) $(SAMPLE_BASE).full

.PHONY: FORCE_QUERY_CONFIG clean-query-config
