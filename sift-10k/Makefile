# Makefile for the SIFT 10k dataset

# --- 1. Dataset-Specific Configuration ---
# All other variables (like filenames and extraction directory)
# are derived automatically by the included recipes.
DATASET_NAME   := sift-10k
ARCHIVE_FILE   := siftsmall.tar.gz
URL            := ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
# --- End of Configuration ---

# --- 2. Include Generic Recipes ---
# Provides the download rule.
include ../common.mk
# Derives all filenames and provides extraction, clean, info, and preview rules.
include ../texmex-fvecs.mk
# --- End of Recipes ---

# --- 3. Define Final Targets ---
all: setup

.PHONY: all