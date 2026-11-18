# recipe-hdf5.mk: Recipe for HDF5 files (which usually don't need decompression).

# The 'setup' target depends on the final data file.
setup: $(FINAL_FILE)

# For HDF5, the downloaded file is the final file.
# We just copy or link it to the expected final name.
$(FINAL_FILE): $(ARCHIVE_FILE)
	@echo "-> Renaming/linking downloaded file..."
	@cp $(ARCHIVE_FILE) $(FINAL_FILE)

# Cleanup rule:
# Removes the data file.
clean:
	@echo "-> Removing data file: $(FINAL_FILE)..."
	rm -f $(FINAL_FILE)

# The full cleanup also triggers this local clean.
clean-all: clean

.PHONY: setup clean