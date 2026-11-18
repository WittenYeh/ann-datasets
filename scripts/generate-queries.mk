# For datasets that provide a full base and no separate query set.
PYTHON ?= python3
num_queries ?= $(NUM_QUERIES)
variance ?= 0.0001

# Deduplicated datasets prepare a raw input first. The others use the full
# source directly as their benchmark base.
FULL_BASE_TARGET := $(or $(RAW_BASE_FILE),$(BASE_FILE))
FULL_BASE_MARKER := $(FULL_BASE_TARGET).full
QUERY_CONFIG := $(QUERY_FILE).config

# An existing installation can have a compacted base newer than its full
# source. Create this marker once to migrate that install on a normal make.
$(FULL_BASE_MARKER):
	@touch "$@"

$(FULL_BASE_TARGET): $(FULL_BASE_FILE) $(FULL_BASE_MARKER)
	@set -eu; tmp=$$(mktemp "$@.XXXXXX"); \
	trap 'rm -f "$$tmp"' EXIT HUP INT TERM; \
	cp --reflink=auto "$<" "$$tmp"; mv -f "$$tmp" "$@"

$(QUERY_CONFIG): FORCE_QUERY_CONFIG
	@set -eu; tmp=$$(mktemp "$@.XXXXXX"); \
	trap 'rm -f "$$tmp"' EXIT HUP INT TERM; \
	printf '%s\n' 'num_queries=$(num_queries)' 'variance=$(variance)' > "$$tmp"; \
	if ! cmp -s "$$tmp" "$@"; then mv -f "$$tmp" "$@"; fi

$(QUERY_FILE): $(BASE_FILE) $(QUERY_CONFIG) ../scripts/generate_queries.py ../scripts/vecs_io.py
	$(PYTHON) ../scripts/generate_queries.py "$<" "$@" \
		--num-queries "$(num_queries)" --variance "$(variance)"

clean: clean-query-config
clean-query-config:
	rm -f $(QUERY_CONFIG) $(FULL_BASE_MARKER)

.PHONY: FORCE_QUERY_CONFIG clean-query-config
