# texmex-bvecs.mk: Recipe for decompressing .gz files.

# The 'setup' target depends on the final, decompressed file.
setup: $(FINAL_FILE)

# Decompression rule:
# Depends on the downloaded archive file.
$(FINAL_FILE): $(ARCHIVE_FILE)
	@echo "-> Decompressing $(ARCHIVE_FILE) with gzip..."
	gzip -d -k -c $(ARCHIVE_FILE) > $(FINAL_FILE)

# Cleanup rule:
# Removes the decompressed file.
clean:
	@echo "-> Removing decompressed file: $(FINAL_FILE)..."
	rm -f $(FINAL_FILE)

# The full cleanup also triggers this local clean.
clean-all: clean

.PHONY: setup clean
