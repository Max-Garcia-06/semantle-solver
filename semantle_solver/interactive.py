"""Interactive hint mode: enter guesses from the game, get the next suggestion."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from semantle_solver.api import normalize_semantle_score
from semantle_solver.suggest import suggest_next_guess
from semantle_solver.word2vec_index import OutOfVocabularyError, Word2VecIndex

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    guess: str
    cosine: float
    remaining: int
    next_guess: str | None
    won: bool


def parse_similarity_input(raw: str) -> float:
    """
    Parse a similarity value from the user.

    Accepts Semantle UI scale (e.g. ``19.20``) or cosine in ``[-1, 1]``.
    """
    value = float(raw.strip())
    if abs(value) > 1.0:
        return normalize_semantle_score(value)
    return value


class HintSession:
    """Maintain the candidate pool and suggest guessable next words."""

    def __init__(self, index: Word2VecIndex, epsilon: float = 1e-4) -> None:
        self.index = index
        self.epsilon = epsilon
        self.candidate_indices = np.arange(index.size, dtype=np.int64)
        self._rejected_indices: set[int] = set()
        self.history: list[tuple[str, float, int]] = []

    def reset(self) -> None:
        self.candidate_indices = np.arange(self.index.size, dtype=np.int64)
        self._rejected_indices.clear()
        self.history.clear()

    def suggest_next(self) -> str | None:
        return suggest_next_guess(
            self.index,
            self.candidate_indices,
            rejected_indices=self._rejected_indices,
        )

    def apply_guess(self, word: str, cosine: float) -> StepResult:
        """
        Filter candidates using the hypersphere for (word, cosine).

        Returns remaining pool size and the suggested next guess.
        """
        try:
            guess_vec = self.index.vector_for(word)
        except OutOfVocabularyError as exc:
            raise OutOfVocabularyError(
                f"{word!r} is not in the Word2Vec vocabulary"
            ) from exc

        before = int(self.candidate_indices.size)
        self.candidate_indices = self.index.filter_by_hypersphere(
            self.candidate_indices,
            guess_vec,
            cosine,
            self.epsilon,
        )
        remaining = int(self.candidate_indices.size)
        self.history.append((word, cosine, remaining))

        logger.info(
            "Filtered %r @ cosine=%.6f: %d -> %d candidates",
            word,
            cosine,
            before,
            remaining,
        )

        won = cosine >= 0.995
        next_guess: str | None = None
        if not won and remaining > 0:
            next_guess = self.suggest_next()
            if next_guess and next_guess.strip().lower() == word.strip().lower():
                try:
                    idx = self.index.word_to_index(word)
                    self._rejected_indices.add(idx)
                except OutOfVocabularyError:
                    pass
                next_guess = self.suggest_next()

        return StepResult(
            guess=word,
            cosine=cosine,
            remaining=remaining,
            next_guess=next_guess,
            won=won,
        )


def run_interactive_loop(session: HintSession) -> int:
    """Read guess/similarity lines from stdin until quit."""
    print()
    print("Semantle hint mode")
    print("  Enter:  <word> <similarity>   (e.g. article 19.20 or article 0.192)")
    print("  Commands:  reset   quit   help")
    print(f"  Candidates: {session.candidate_indices.size:,}  |  epsilon: {session.epsilon}")
    suggested = session.suggest_next()
    if suggested:
        print(f"  First suggested guess: {suggested}")
    print()

    while True:
        try:
            line = input("guess> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not line:
            continue

        lower = line.lower()
        if lower in {"quit", "exit", "q"}:
            return 0
        if lower == "reset":
            session.reset()
            print(f"Reset. {session.candidate_indices.size:,} candidates.")
            suggested = session.suggest_next()
            if suggested:
                print(f"Suggested next guess: {suggested}")
            continue
        if lower == "help":
            print(
                "\n  Enter:  <word> <similarity>\n"
                "  Commands:  reset   quit\n"
                f"  Candidates: {session.candidate_indices.size:,}\n"
            )
            continue

        parts = line.split()
        if len(parts) < 2:
            print("Usage: <word> <similarity>")
            continue

        word = parts[0]
        try:
            cosine = parse_similarity_input(parts[1])
        except ValueError:
            print(f"Invalid similarity: {parts[1]!r}")
            continue

        try:
            result = session.apply_guess(word, cosine)
        except OutOfVocabularyError as exc:
            print(f"Error: {exc}")
            continue

        if result.won:
            print(f"Similarity ~100 — you likely found it: {word!r}")
            return 0

        if result.remaining == 0:
            print("No candidates left. Try a larger --epsilon or recheck the similarity.")
            return 1

        if result.remaining == 1 and result.next_guess:
            print(f"One candidate left — likely answer: {result.next_guess!r}")
        else:
            print(f"{result.remaining:,} candidates remain.")

        if result.next_guess:
            print(f"Suggested next guess: {result.next_guess}")
        else:
            print("No guessable candidate found in the pool.")
