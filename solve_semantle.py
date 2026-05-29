#!/usr/bin/env python3
"""
Semantle solver via hypersphere intersection in Word2Vec space.

Example:
    python solve_semantle.py
    python solve_semantle.py --game-id 1611 --initial-guess country
    python solve_semantle.py --epsilon 1e-3 -v
"""

from __future__ import annotations

import argparse
import logging
import sys

from semantle_solver.api import SemantleAPIError, current_puzzle_date, game_id_for_date
from semantle_solver.model_loader import ModelDownloadError
from semantle_solver.solver import HypersphereSolver, SolverConfig
from semantle_solver.word2vec_index import OutOfVocabularyError, Word2VecIndex


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Solve Semantle using hypersphere intersection over Word2Vec."
    )
    parser.add_argument(
        "--initial-guess",
        default="article",
        help="First guess word (default: article)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-4,
        help="Cosine tolerance for hypersphere filter (default: 1e-4)",
    )
    parser.add_argument(
        "--max-guesses",
        type=int,
        default=200,
        help="Safety cap on API guesses (default: 200)",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Puzzle language code (default: en)",
    )
    parser.add_argument(
        "--game-id",
        type=int,
        default=None,
        help="Puzzle id (default: inferred from today's NY date)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Hint mode: enter each guess and similarity, get the next suggestion",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    if args.verbose:
        # Keep -v useful without dumping every HTTP redirect line.
        logging.getLogger("urllib3").setLevel(logging.WARNING)

    if args.interactive:
        from semantle_solver.interactive import HintSession, run_interactive_loop

        print("Loading Word2Vec embeddings (cached after first run)...")
        try:
            index = Word2VecIndex.from_gensim()
            session = HintSession(index=index, epsilon=args.epsilon)
        except (ModelDownloadError, RuntimeError) as exc:
            logging.error("%s", exc)
            return 1
        return run_interactive_loop(session)

    puzzle_day = current_puzzle_date()
    default_gid = game_id_for_date(puzzle_day)
    game_id = args.game_id if args.game_id is not None else default_gid
    logging.info("Puzzle date (NY): %s  game_id=%s", puzzle_day, game_id)

    try:
        index = Word2VecIndex.from_gensim()
        config = SolverConfig(
            initial_guess=args.initial_guess,
            epsilon=args.epsilon,
            max_guesses=args.max_guesses,
            language=args.language,
            game_id=game_id,
        )
        result = HypersphereSolver(index=index, config=config).solve()
    except (SemantleAPIError, OutOfVocabularyError, RuntimeError, ModelDownloadError) as exc:
        logging.error("%s", exc)
        return 1

    print("\n=== Solve summary ===")
    print(f"Secret word: {result.secret_word}")
    print(f"Victory: {result.victory}")
    print(f"Guesses: {len(result.guesses)}")
    for i, row in enumerate(result.guesses, start=1):
        print(
            f"  {i}. {row.guess:20s}  cosine={row.cosine: .6f}  "
            f"(api={row.api_score:7.3f})  pool={row.remaining}"
        )
    return 0 if result.victory else 2


if __name__ == "__main__":
    sys.exit(main())
