"""Parse gensim-data word2vec C-binary (space-delimited records, not null-terminated)."""

from __future__ import annotations

import gzip
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

import numpy as np

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1024 * 1024


def _read_header(stream: BinaryIO) -> tuple[int, int]:
    header = stream.readline()
    parts = header.decode("utf-8", errors="replace").strip().split()
    if len(parts) != 2:
        raise ValueError(f"unexpected word2vec header: {header!r}")
    return int(parts[0]), int(parts[1])


def _iter_records(
    stream: BinaryIO, vocab_size: int, vector_size: int
) -> Iterator[tuple[str, np.ndarray]]:
    """
    Yield (word, vector) using the same record layout as gensim's binary loader.

    Each record is: ``WORD<space>float32×vector_size`` (no null byte between records).
    """
    bytes_per_vector = vector_size * 4
    chunk = b""
    yielded = 0

    while yielded < vocab_size:
        if len(chunk) < bytes_per_vector + 64:
            new_data = stream.read(CHUNK_SIZE)
            if new_data:
                chunk += new_data
            elif len(chunk) < bytes_per_vector + 1:
                break

        space_idx = chunk.find(b" ")
        if space_idx == -1 or len(chunk) - (space_idx + 1) < bytes_per_vector:
            new_data = stream.read(CHUNK_SIZE)
            if not new_data:
                break
            chunk += new_data
            continue

        word = chunk[:space_idx].decode("utf-8", errors="replace").lstrip("\n")
        vector_start = space_idx + 1
        vector_end = vector_start + bytes_per_vector
        vector = np.frombuffer(
            chunk[vector_start:vector_end], dtype=np.float32
        ).copy()
        chunk = chunk[vector_end:]
        yielded += 1
        yield word, vector

    if yielded != vocab_size:
        raise EOFError(
            f"expected {vocab_size} vectors, parsed {yielded} "
            "(file may be truncated or format changed)"
        )


def iter_word2vec_gz(path: Path) -> Iterator[tuple[str, np.ndarray]]:
    """Stream word/vector pairs from a .gz word2vec binary file."""
    with gzip.open(path, "rb") as handle:
        vocab_size, vector_size = _read_header(handle)
        yield from _iter_records(handle, vocab_size, vector_size)


def sanity_check_vectors(path: Path) -> bool:
    """
    Return True if the archive parses and contains plausible vectors.

    The first token (``</s>``) has a near-zero vector; we spot-check common words
    instead of relying on the leading record.
    """
    try:
        for index, (word, vector) in enumerate(iter_word2vec_gz(path)):
            if not bool(np.isfinite(vector).all()):
                continue
            norm = float(np.linalg.norm(vector))
            token = word.strip()
            if token in {"article", "cover", "king", "woman"}:
                return 0.5 < norm < 5.0
            if index > 50_000:
                break
        return False
    except (OSError, EOFError, ValueError):
        return False
