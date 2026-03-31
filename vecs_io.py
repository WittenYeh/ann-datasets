#!/usr/bin/env python3
"""
Shared I/O utilities for fvecs / bvecs / ivecs files.

Provides both memory-mapped (zero-copy) and full-read variants,
plus vectorized write functions.  Aligned with the conventions from
  - https://github.com/matsui528/deep1b_gt
  - https://github.com/facebookresearch/faiss/blob/master/contrib/vecs_io.py
"""

import numpy as np

# ── memory-mapped readers (zero-copy, ideal for large files) ─────────────

def ivecs_mmap(fname):
    """Memory-map an ivecs file. Returns a read-only (n, d) int32 view."""
    a = np.memmap(fname, dtype="int32", mode="r")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:]


def fvecs_mmap(fname):
    """Memory-map an fvecs file. Returns a read-only (n, d) float32 view."""
    return ivecs_mmap(fname).view("float32")


def bvecs_mmap(fname):
    """Memory-map a bvecs file. Returns a read-only (n, d) uint8 view."""
    a = np.memmap(fname, dtype="uint8", mode="r")
    d = int.from_bytes(a[:4].tobytes(), byteorder="little")
    return a.reshape(-1, d + 4)[:, 4:]


# ── full readers (load everything into RAM, suitable for small files) ────

def ivecs_read(fname):
    """Read an entire ivecs file into a contiguous (n, d) int32 array."""
    a = np.fromfile(fname, dtype="int32")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    """Read an entire fvecs file into a contiguous (n, d) float32 array."""
    return ivecs_read(fname).view("float32")


def bvecs_read(fname):
    """Read an entire bvecs file into a contiguous (n, d) uint8 array."""
    a = np.memmap(fname, dtype="uint8", mode="r")
    d = int.from_bytes(a[:4].tobytes(), byteorder="little")
    return np.array(a.reshape(-1, d + 4)[:, 4:], dtype="uint8")


# ── writers (vectorized, no Python loops) ────────────────────────────────

def ivecs_write(fname, m):
    """Write an (n, d) int32 array in ivecs format."""
    n, d = m.shape
    m1 = np.empty((n, d + 1), dtype="int32")
    m1[:, 0] = d
    m1[:, 1:] = m
    m1.tofile(fname)


def fvecs_write(fname, m):
    """Write an (n, d) float32 array in fvecs format."""
    m = np.asarray(m, dtype="float32")
    ivecs_write(fname, m.view("int32"))


def bvecs_write(fname, m):
    """Write an (n, d) uint8 array in bvecs format."""
    m = np.asarray(m, dtype="uint8")
    n, d = m.shape
    m1 = np.empty((n, d + 4), dtype="uint8")
    m1[:, :4] = np.array([d], dtype="int32").view("uint8")
    m1[:, 4:] = m
    m1.tofile(fname)


# ── helpers ──────────────────────────────────────────────────────────────

def mmap_by_ext(fname):
    """Auto-select mmap reader based on file extension."""
    ext = fname.rsplit(".", 1)[-1].lower()
    if ext == "fvecs":
        return fvecs_mmap(fname)
    elif ext == "bvecs":
        return bvecs_mmap(fname)
    elif ext == "ivecs":
        return ivecs_mmap(fname)
    raise ValueError(f"Unsupported extension: .{ext}")


def read_by_ext(fname):
    """Auto-select full reader based on file extension."""
    ext = fname.rsplit(".", 1)[-1].lower()
    if ext == "fvecs":
        return fvecs_read(fname)
    elif ext == "bvecs":
        return bvecs_read(fname)
    elif ext == "ivecs":
        return ivecs_read(fname)
    raise ValueError(f"Unsupported extension: .{ext}")


def write_by_ext(fname, m):
    """Auto-select writer based on file extension."""
    ext = fname.rsplit(".", 1)[-1].lower()
    if ext == "fvecs":
        fvecs_write(fname, m)
    elif ext == "bvecs":
        bvecs_write(fname, m)
    elif ext == "ivecs":
        ivecs_write(fname, m)
    else:
        raise ValueError(f"Unsupported extension: .{ext}")
