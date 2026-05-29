"""Shared next-guess selection for auto-solve and interactive hint modes."""

from __future__ import annotations

import numpy as np

from semantle_solver.api import is_guessable_word
from semantle_solver.word2vec_index import Word2VecIndex


def suggest_next_guess(
    index: Word2VecIndex,
    candidate_indices: np.ndarray,
    *,
    rejected_indices: set[int] | None = None,
) -> str | None:
    """Pick a centroid-near, Semantle-friendly word from the candidate pool."""
    if candidate_indices.size == 0:
        return None
    if candidate_indices.size == 1:
        return index.vocabulary[int(candidate_indices[0])]

    rejected = rejected_indices or set()
    ranked = index.rank_candidate_indices(candidate_indices)

    for idx in ranked:
        idx = int(idx)
        if idx in rejected:
            continue
        word = index.vocabulary[idx]
        if is_guessable_word(word):
            return word

    for idx in ranked:
        idx = int(idx)
        if idx not in rejected:
            return index.vocabulary[idx]
    return None
