"""Word2Vec vocabulary matrix with L2-normalized rows for cosine dot products."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from semantle_solver.model_loader import MODEL_NAME, load_vocabulary_matrix

logger = logging.getLogger(__name__)


class OutOfVocabularyError(LookupError):
    """Raised when a token is not in the Word2Vec vocabulary."""


@dataclass
class Word2VecIndex:
    """Dense matrix V where row i is the unit vector for vocabulary[i]."""

    vocabulary: list[str]
    vectors: np.ndarray  # shape (N, 300), float32, L2-normalized rows
    _word_to_idx: dict[str, int]

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def size(self) -> int:
        return int(self.vectors.shape[0])

    @classmethod
    def from_gensim(cls, model_name: str = MODEL_NAME) -> Word2VecIndex:
        logger.info("Loading Word2Vec model %s (first run downloads ~1.6GB)...", model_name)
        vocab, matrix = load_vocabulary_matrix()
        # Binary fallback normalizes during parse; gensim path normalizes in loader.
        if not isinstance(matrix, np.memmap):
            logger.info("Normalizing %d vocabulary tokens...", len(vocab))
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms = np.where(norms == 0.0, 1.0, norms)
            matrix = matrix / norms
        word_to_idx = {w: i for i, w in enumerate(vocab)}
        return cls(vocabulary=vocab, vectors=matrix, _word_to_idx=word_to_idx)

    def _lookup_candidates(self, word: str) -> list[str]:
        """
        Build lookup keys for Google News Word2Vec.

        Most single-token entries are stored with a trailing space (e.g. ``article ``).
        Phrases use underscores (e.g. ``New_York_City``).
        """
        stripped = word.strip()
        if not stripped:
            return []
        underscored = stripped.replace(" ", "_")
        underscored_lower = underscored.lower()
        return [
            word,
            stripped,
            f"{stripped} ",
            stripped.lower(),
            f"{stripped.lower()} ",
            underscored,
            f"{underscored} ",
            underscored_lower,
            f"{underscored_lower} ",
        ]

    def word_to_index(self, word: str) -> int:
        seen: set[str] = set()
        for candidate in self._lookup_candidates(word):
            if candidate in seen:
                continue
            seen.add(candidate)
            idx = self._word_to_idx.get(candidate)
            if idx is not None:
                return idx
        raise OutOfVocabularyError(f"{word!r} is not in the Word2Vec vocabulary")

    def vector_for(self, word: str) -> np.ndarray:
        idx = self.word_to_index(word)
        return self.vectors[idx]

    def filter_by_hypersphere(
        self,
        indices: np.ndarray,
        guess_vector: np.ndarray,
        target_cosine: float,
        epsilon: float,
    ) -> np.ndarray:
        """
        Retain candidate indices where |cos(guess, w) - target_cosine| < epsilon.

        Uses vectorized matmul: sim = V_sub @ g with unit-norm rows/columns.
        """
        if indices.size == 0:
            return indices
        g = np.asarray(guess_vector, dtype=np.float32).reshape(-1)
        g_norm = np.linalg.norm(g)
        if g_norm == 0.0:
            raise ValueError("guess vector has zero norm")
        g = g / g_norm
        block = self.vectors[indices]
        similarities = block @ g
        mask = np.abs(similarities - target_cosine) < epsilon
        return indices[mask]

    def rank_candidate_indices(self, indices: np.ndarray) -> np.ndarray:
        """Return candidate indices sorted by similarity to their centroid (best first)."""
        if indices.size <= 1:
            return indices
        block = self.vectors[indices]
        centroid = block.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm == 0.0:
            return indices
        centroid /= centroid_norm
        scores = block @ centroid
        order = np.argsort(-scores)
        return indices[order]

    def words_for_indices(self, indices: Sequence[int]) -> list[str]:
        return [self.vocabulary[i] for i in indices]
