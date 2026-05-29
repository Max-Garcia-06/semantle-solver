"""Load word2vec-google-news-300 (gensim downloader or direct binary fallback)."""

from __future__ import annotations

import gzip
import importlib.util
import logging
from pathlib import Path

import numpy as np
import requests

from semantle_solver.word2vec_binary import iter_word2vec_gz, sanity_check_vectors

logger = logging.getLogger(__name__)

MODEL_NAME = "word2vec-google-news-300"
MODEL_GZ_URL = (
    "https://github.com/RaRe-Technologies/gensim-data/releases/download/"
    "word2vec-google-news-300/word2vec-google-news-300.gz"
)
EXPECTED_BYTES = 1_743_563_840
VOCAB_SIZE = 3_000_000
VECTOR_DIM = 300

CACHE_DIR = Path.home() / ".cache" / "semantle-solver"
CACHE_GZ = CACHE_DIR / "word2vec-google-news-300.gz"
# v2: correct space-delimited binary parse (v1 used wrong null-byte layout).
CACHE_VECTORS = CACHE_DIR / "vectors-v2.npy"
CACHE_VOCAB = CACHE_DIR / "vocabulary.bin"
LEGACY_CACHE_VOCAB = CACHE_DIR / "vocabulary.txt"


class ModelDownloadError(RuntimeError):
    """Raised when the embedding archive cannot be downloaded or verified."""


def _gensim_available() -> bool:
    return importlib.util.find_spec("gensim") is not None


def _load_via_gensim() -> tuple[list[str], np.ndarray]:
    from gensim import downloader as api  # type: ignore[import-untyped]

    model = api.load(MODEL_NAME)
    vocab = list(model.index_to_key)
    dim = model.vector_size
    matrix = np.zeros((len(vocab), dim), dtype=np.float32)
    for i, word in enumerate(vocab):
        matrix[i] = model.get_vector(word)
    return vocab, matrix


def _remote_content_length() -> int | None:
    try:
        response = requests.head(MODEL_GZ_URL, allow_redirects=True, timeout=30)
        if response.ok and response.headers.get("Content-Length"):
            return int(response.headers["Content-Length"])
    except requests.RequestException as exc:
        logger.debug("Could not read Content-Length: %s", exc)
    return None


def _gz_on_disk_is_complete(path: Path, expected_bytes: int) -> bool:
    return path.is_file() and path.stat().st_size >= expected_bytes * 0.99


def _is_valid_gz(path: Path, expected_bytes: int) -> bool:
    if not _gz_on_disk_is_complete(path, expected_bytes):
        if path.is_file():
            logger.warning(
                "Cached model is incomplete (%d / %d bytes).",
                path.stat().st_size,
                expected_bytes,
            )
        return False
    try:
        with gzip.open(path, "rb") as handle:
            header = handle.readline()
            parts = header.decode("utf-8", errors="replace").strip().split()
            if len(parts) != 2:
                return False
            vocab_size, dim = int(parts[0]), int(parts[1])
            return vocab_size == VOCAB_SIZE and dim == VECTOR_DIM
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        logger.warning("Cached model failed gzip/header check: %s", exc)
        return False


def _download_model_gz(destination: Path, expected_bytes: int) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    remote_len = _remote_content_length()
    target_bytes = remote_len or expected_bytes

    if _is_valid_gz(destination, target_bytes):
        return destination

    if destination.exists():
        logger.info("Removing incomplete cache: %s", destination)
        destination.unlink()
    partial.unlink(missing_ok=True)

    logger.info(
        "Downloading %s (~%.1f GB). This can take several minutes...",
        MODEL_GZ_URL,
        target_bytes / (1024**3),
    )
    written = 0
    try:
        with requests.get(
            MODEL_GZ_URL,
            stream=True,
            timeout=(30, 120),
        ) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    written += len(chunk)
                    if written % (100 * 1024 * 1024) < len(chunk):
                        logger.info(
                            "  ... %.1f / %.1f GB",
                            written / (1024**3),
                            target_bytes / (1024**3),
                        )
    except requests.RequestException as exc:
        partial.unlink(missing_ok=True)
        raise ModelDownloadError(f"download failed: {exc}") from exc

    if written < target_bytes * 0.99:
        partial.unlink(missing_ok=True)
        raise ModelDownloadError(
            f"download incomplete ({written} bytes, expected ~{target_bytes}). "
            "Check your network and retry."
        )

    partial.rename(destination)
    if not _is_valid_gz(destination, target_bytes):
        destination.unlink(missing_ok=True)
        raise ModelDownloadError("downloaded file failed integrity check")
    logger.info("Download complete (%d bytes).", destination.stat().st_size)
    return destination


def _read_vocabulary_file(path: Path) -> list[str]:
    """Read null-separated UTF-8 tokens (tokens may contain spaces/newlines)."""
    raw = path.read_bytes()
    if not raw:
        return []
    parts = raw.split(b"\0")
    if parts and parts[-1] == b"":
        parts = parts[:-1]
    return [part.decode("utf-8", errors="replace") for part in parts]


def _normalize_memmap_in_chunks(matrix: np.memmap, chunk_size: int = 50_000) -> None:
    rows = matrix.shape[0]
    for start in range(0, rows, chunk_size):
        end = min(start + chunk_size, rows)
        block = np.asarray(matrix[start:end], dtype=np.float64)
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        matrix[start:end] = (block / norms).astype(np.float32)
        matrix.flush()


def _parse_word2vec_gz_to_disk(gz_path: Path) -> None:
    """
    Stream .gz → memmap matrix + vocabulary file.

    Uses disk-backed memmap so peak RAM stays well below 3.6 GB.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_vectors = CACHE_VECTORS.with_suffix(".npy.part")
    tmp_vocab = CACHE_VOCAB.with_suffix(".bin.part")
    tmp_vectors.unlink(missing_ok=True)
    tmp_vocab.unlink(missing_ok=True)

    matrix = np.lib.format.open_memmap(
        tmp_vectors,
        mode="w+",
        dtype=np.float32,
        shape=(VOCAB_SIZE, VECTOR_DIM),
    )

    if not sanity_check_vectors(gz_path):
        raise ModelDownloadError(f"{gz_path} failed embedding sanity check")

    word_index = -1
    with tmp_vocab.open("wb") as vocab_file:
        for word_index, (word, vector) in enumerate(iter_word2vec_gz(gz_path)):
            if word_index >= VOCAB_SIZE:
                break
            vocab_file.write(word.encode("utf-8") + b"\0")
            matrix[word_index] = vector
            if (word_index + 1) % 500_000 == 0:
                matrix.flush()
                logger.info("  ... parsed %d / %d vectors", word_index + 1, VOCAB_SIZE)

    if word_index + 1 != VOCAB_SIZE:
        raise EOFError(f"expected {VOCAB_SIZE} records, parsed {word_index + 1}")

    logger.info("L2-normalizing rows (chunked)...")
    _normalize_memmap_in_chunks(matrix)
    matrix.flush()
    tmp_vectors.rename(CACHE_VECTORS)
    tmp_vocab.rename(CACHE_VOCAB)
    logger.info("Saved parsed cache to %s", CACHE_DIR)


def _cache_matches_api(matrix: np.ndarray, vocab: list[str]) -> bool:
    """Verify local vectors match Semantle server cosine for a spot-check pair."""
    try:
        from semantle_solver.api import (
            SemantleClient,
            current_puzzle_date,
            game_id_for_date,
            normalize_semantle_score,
        )

        def row_for(token: str) -> int | None:
            for form in (token, f"{token} ", token.replace(" ", "_")):
                try:
                    return vocab.index(form)
                except ValueError:
                    continue
            return None

        article_row = row_for("article")
        cover_row = row_for("cover")
        if article_row is None or cover_row is None:
            return True
        client = SemantleClient()
        game = client.fetch_game(game_id_for_date(current_puzzle_date()))
        api_cosine = normalize_semantle_score(
            client.submit_guess("article", game.secret_word).initial_similarity
        )
        local = float(matrix[article_row] @ matrix[cover_row])
        return abs(local - api_cosine) < 0.01
    except Exception as exc:
        logger.debug("API parity check skipped: %s", exc)
        return True


def _load_parsed_cache() -> tuple[list[str], np.ndarray] | None:
    if not CACHE_VECTORS.is_file():
        return None
    try:
        matrix = np.load(CACHE_VECTORS, mmap_mode="r", allow_pickle=False)
        if tuple(matrix.shape) != (VOCAB_SIZE, VECTOR_DIM):
            logger.warning("Stale vector cache; rebuilding from .gz")
            return None
        vocab: list[str] | None = None
        if CACHE_VOCAB.is_file():
            vocab = _read_vocabulary_file(CACHE_VOCAB)
        if vocab is None or len(vocab) != int(matrix.shape[0]):
            return None
        if not _cache_matches_api(matrix, vocab):
            logger.warning("Vector cache failed API parity check; rebuilding from .gz")
            return None
        logger.info("Loaded parsed cache from %s (mmap)", CACHE_DIR)
        return vocab, matrix
    except (OSError, ValueError) as exc:
        logger.warning("Could not read parsed cache (%s); rebuilding.", exc)
        return None


def _load_via_binary_download() -> tuple[list[str], np.ndarray]:
    cached = _load_parsed_cache()
    if cached is not None:
        return cached

    gz_path = _download_model_gz(CACHE_GZ, EXPECTED_BYTES)
    try:
        _parse_word2vec_gz_to_disk(gz_path)
    except MemoryError as exc:
        raise ModelDownloadError(
            "Ran out of memory while loading embeddings. Close other apps and retry. "
            "If parsing was interrupted, delete vectors.npy.part in the cache folder."
        ) from exc
    except EOFError as exc:
        if _gz_on_disk_is_complete(gz_path, EXPECTED_BYTES):
            raise ModelDownloadError(
                f"Failed parsing a complete download ({gz_path}): {exc}. "
                "Free RAM and retry — the .gz will not be re-downloaded."
            ) from exc
        gz_path.unlink(missing_ok=True)
        raise ModelDownloadError(
            f"Corrupt embedding file removed ({gz_path}). Re-run to download again."
        ) from exc

    loaded = _load_parsed_cache()
    if loaded is None:
        raise ModelDownloadError("parse finished but cache files are missing")
    return loaded


def load_vocabulary_matrix() -> tuple[list[str], np.ndarray]:
    """
    Return (vocabulary, L2-normalized embedding matrix).

    Prefer gensim.downloader when installed; otherwise stream the official
    gensim-data gzip and parse the binary format with NumPy.
    """
    if _gensim_available():
        try:
            logger.info("Loading via gensim.downloader.load(%s)...", MODEL_NAME)
            vocab, matrix = _load_via_gensim()
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms = np.where(norms == 0.0, 1.0, norms)
            matrix = matrix / norms
            return vocab, matrix
        except Exception as exc:  # noqa: BLE001
            logger.warning("gensim load failed (%s); using binary fallback.", exc)
    else:
        logger.info("gensim not installed; using binary fallback.")
    return _load_via_binary_download()
