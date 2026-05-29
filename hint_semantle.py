#!/usr/bin/env python3
"""
Interactive Semantle assistant: enter your guess + similarity, get the next guess.

Example:
    python hint_semantle.py
    python hint_semantle.py --epsilon 1e-4
"""

from __future__ import annotations

import argparse
import logging
import sys

from semantle_solver.interactive import HintSession, run_interactive_loop
from semantle_solver.model_loader import ModelDownloadError
from semantle_solver.word2vec_index import Word2VecIndex


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Semantle hint mode: filter candidates from your guess and similarity."
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-4,
        help="Cosine tolerance for hypersphere filter (default: 1e-4)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    print("Loading Word2Vec embeddings (cached after first run)...")
    try:
        index = Word2VecIndex.from_gensim()
        session = HintSession(index=index, epsilon=args.epsilon)
    except (ModelDownloadError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return run_interactive_loop(session)


if __name__ == "__main__":
    sys.exit(main())
