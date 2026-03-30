# Makefile for the SIFT 10k dataset

# --- 1. Dataset-Specific Configuration ---
DATASET_NAME   := sift-10k
ARCHIVE_FILE   := siftsmall.tar.gz
URL            := ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
# --- End of Configuration ---

# --- 2. Define Default Target (must be before includes) ---
all: setup
.PHONY: all

# --- 3. Include Generic Recipes ---
include ../common.mk
include ../texmex-fvecs.mk
