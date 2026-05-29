"""Hypersphere-intersection Semantle solver."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from semantle_solver.api import (
    GuessNotFoundError,
    SemantleAPIError,
    SemantleClient,
)
from semantle_solver.suggest import suggest_next_guess
from semantle_solver.word2vec_index import OutOfVocabularyError, Word2VecIndex

logger = logging.getLogger(__name__)


@dataclass
class SolverConfig:
    initial_guess: str = "article"
    epsilon: float = 1e-4
    max_guesses: int = 200
    language: str = "en"
    game_id: int | None = None  # None => today's puzzle (NY timezone)


@dataclass
class GuessRecord:
    guess: str
    cosine: float
    api_score: float
    remaining: int


@dataclass
class SolveResult:
    secret_word: str
    guesses: list[GuessRecord] = field(default_factory=list)
    victory: bool = False


class HypersphereSolver:
    """
    Isolate the secret embedding by intersecting cosine hyperspheres.

    Each guess g with server cosine c_t eliminates any candidate w with
    cos(g, w) not matching c_t within tolerance epsilon.
    """

    def __init__(
        self,
        index: Word2VecIndex,
        client: SemantleClient | None = None,
        config: SolverConfig | None = None,
    ) -> None:
        self.index = index
        self.client = client or SemantleClient(language=(config or SolverConfig()).language)
        self.config = config or SolverConfig()
        self._rejected_indices: set[int] = set()

    def _resolve_guess(
        self, candidate_indices: np.ndarray, preferred: str | None
    ) -> tuple[str, int]:
        if preferred is not None:
            try:
                idx = self.index.word_to_index(preferred)
                if (
                    np.any(candidate_indices == idx)
                    and idx not in self._rejected_indices
                ):
                    return preferred, idx
            except OutOfVocabularyError:
                pass

        word = suggest_next_guess(
            self.index,
            candidate_indices,
            rejected_indices=self._rejected_indices,
        )
        if word is None:
            raise RuntimeError("no API-acceptable guess remaining in candidate pool")
        return word, self.index.word_to_index(word)

    def solve(self) -> SolveResult:
        cfg = self.config
        self.client.language = cfg.language
        self._rejected_indices.clear()

        try:
            game = self.client.fetch_game(cfg.game_id)
        except SemantleAPIError as exc:
            raise SemantleAPIError(f"could not load game: {exc}") from exc

        logger.info(
            "Puzzle #%s (%s); filtering %d embeddings (secret held server-side for API only).",
            game.game_id,
            game.language,
            self.index.size,
        )

        candidate_indices = np.arange(self.index.size, dtype=np.int64)
        records: list[GuessRecord] = []
        pending_guess: str | None = cfg.initial_guess
        skipped_guesses = 0
        max_skips = 500

        while len(records) < cfg.max_guesses:
            if candidate_indices.size == 0:
                raise RuntimeError("candidate pool empty before solution")
            if skipped_guesses >= max_skips:
                raise RuntimeError("too many API-rejected guesses; try increasing epsilon")

            guess_word, guess_idx = self._resolve_guess(candidate_indices, pending_guess)
            pending_guess = None

            try:
                guess_vec = self.index.vector_for(guess_word)
            except OutOfVocabularyError:
                self._rejected_indices.add(guess_idx)
                skipped_guesses += 1
                logger.warning("Skipping OOV guess %r", guess_word)
                continue

            try:
                api_result = self.client.submit_guess(guess_word, game.secret_word)
            except GuessNotFoundError:
                self._rejected_indices.add(guess_idx)
                skipped_guesses += 1
                logger.warning(
                    "Semantle rejected %r (not in game dictionary); trying another candidate",
                    guess_word,
                )
                continue
            except SemantleAPIError as exc:
                raise SemantleAPIError(f"guess {guess_word!r} failed: {exc}") from exc

            target_cosine = api_result.cosine
            records.append(
                GuessRecord(
                    guess=guess_word,
                    cosine=target_cosine,
                    api_score=api_result.initial_similarity,
                    remaining=int(candidate_indices.size),
                )
            )

            logger.info(
                "Guess %d: %r  cosine=%.6f  (api=%.4f)  candidates=%d",
                len(records),
                guess_word,
                target_cosine,
                api_result.initial_similarity,
                candidate_indices.size,
            )

            if api_result.is_victory or guess_word.strip().lower() == game.secret_word:
                return SolveResult(
                    secret_word=game.secret_word,
                    guesses=records,
                    victory=True,
                )

            candidate_indices = self.index.filter_by_hypersphere(
                candidate_indices,
                guess_vec,
                target_cosine,
                cfg.epsilon,
            )

            logger.info("After filter: %d candidates remain", candidate_indices.size)

            if candidate_indices.size == 1:
                only_idx = int(candidate_indices[0])
                pending_guess = self.index.vocabulary[only_idx]
                continue

            if candidate_indices.size == 0:
                raise RuntimeError(
                    f"no candidates remain after guess {guess_word!r}; "
                    "try increasing epsilon or check embedding parity"
                )

        raise RuntimeError(f"exceeded max_guesses={cfg.max_guesses} without solving")
