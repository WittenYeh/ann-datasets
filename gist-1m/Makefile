# Makefile for the GIST 1 Million dataset

# --- 1. Dataset-Specific Configuration ---
DATASET_NAME   := gist-1m
ARCHIVE_FILE   := gist.tar.gz
URL            := ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz
# --- End of Configuration ---

# --- 2. Define Default Target (must be before includes) ---
all: setup
.PHONY: all

# --- 3. Include Generic Recipes ---
include ../common.mk
include ../texmex-fvecs.mk
