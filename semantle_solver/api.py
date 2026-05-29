"""HTTP client for server.semantle.com (Semantle backend)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import requests
from requests import Response, Session

logger = logging.getLogger(__name__)

NY_TZ = ZoneInfo("America/New_York")
BASE_URL = "https://server.semantle.com"
DEFAULT_TIMEOUT = 30


class SemantleAPIError(RuntimeError):
    """Raised when the Semantle API returns an error or unexpected payload."""


class GuessNotFoundError(SemantleAPIError):
    """Raised when the server does not accept a guess word (HTTP 404)."""


@dataclass(frozen=True)
class GuessResult:
    guess: str
    similarity: float
    initial_similarity: float
    percentile: int | None
    closest_similarity: float
    raw: dict[str, Any]

    @property
    def cosine(self) -> float:
        """Cosine similarity in [-1, 1] derived from the stable server score."""
        return normalize_semantle_score(self.initial_similarity)

    @property
    def is_victory(self) -> bool:
        return self.initial_similarity >= 99.5 or self.similarity >= 99.5


@dataclass(frozen=True)
class GameInfo:
    game_id: int
    language: str
    secret_word: str
    similarity_closest: float
    similarity_tenth: float
    similarity_thousandth: float


def is_guessable_word(word: str) -> bool:
    """
    Heuristic for tokens Semantle is likely to accept.

    The full Word2Vec vocab includes many proper-noun phrases the game API rejects.
    """
    token = word.strip()
    if len(token) < 2 or len(token) > 32:
        return False
    if token.startswith(("</", "'")):
        return False
    if any(ch.isupper() for ch in token):
        return False
    core = token.replace("_", "").replace("-", "")
    if not core.isalpha() or not core.islower():
        return False
    if token.count("_") > 2:
        return False
    return True


def normalize_semantle_score(score: float) -> float:
    """
    Map UI / API similarity to cosine similarity in [-1, 1].

    Semantle displays cosine similarity multiplied by 100.
    """
    return float(score) / 100.0


def current_puzzle_date(now: datetime | None = None) -> date:
    """Match Semantle's America/New_York rollover (new word at 8 PM ET)."""
    instant = now or datetime.now(NY_TZ)
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=NY_TZ)
    else:
        instant = instant.astimezone(NY_TZ)
    puzzle_day = instant.date()
    if instant.hour < 20:
        puzzle_day -= timedelta(days=1)
    return puzzle_day


def game_id_for_date(puzzle_day: date) -> int:
    """
    Replicate the client's generatePerms() id assignment.

    IDs start at 1 for 2021-12-29 and increment by one calendar day.
    """
    origin = date(2021, 12, 29)
    if puzzle_day < origin:
        raise ValueError(f"date {puzzle_day} is before Semantle puzzle origin {origin}")
    return (puzzle_day - origin).days + 1


class SemantleClient:
    """
    Interact with the Semantle game server.

    The public similarity endpoint is:

        GET /similarity/{guess}/{secret_word}/{language}

    Scores in the JSON response are cosine similarity × 100. Use
    ``initialSimilarity`` for filtering (it is stable; ``similarity`` may animate).
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
        language: str = "en",
        timeout: float = DEFAULT_TIMEOUT,
        session: Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.language = language
        self.timeout = timeout
        self.session = session or requests.Session()

    def _request(self, method: str, path: str, **kwargs: Any) -> Response:
        url = f"{self.base_url}{path}"
        try:
            response = self.session.request(
                method,
                url,
                timeout=self.timeout,
                **kwargs,
            )
        except requests.RequestException as exc:
            raise SemantleAPIError(f"network error for {url}: {exc}") from exc
        if not response.ok:
            body = response.text[:500]
            if response.status_code == 404 and "not found" in body.lower():
                raise GuessNotFoundError(
                    f"{method} {url} rejected guess ({response.status_code}): {body}"
                )
            raise SemantleAPIError(
                f"{method} {url} failed ({response.status_code}): {body}"
            )
        return response

    def fetch_game(self, game_id: int | None = None) -> GameInfo:
        gid = game_id if game_id is not None else game_id_for_date(current_puzzle_date())
        path = f"/semantle/game/{gid}/{self.language}"
        response = self._request("GET", path)
        try:
            payload = response.json()
        except ValueError as exc:
            raise SemantleAPIError("game response is not valid JSON") from exc
        try:
            return GameInfo(
                game_id=gid,
                language=self.language,
                secret_word=str(payload["secretWord"]).lower(),
                similarity_closest=float(payload["similarityClosest"]),
                similarity_tenth=float(payload["similarityTenth"]),
                similarity_thousandth=float(payload["similarityThousandth"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SemantleAPIError(f"unexpected game payload: {payload!r}") from exc

    def submit_guess(self, guess: str, secret_word: str) -> GuessResult:
        """
        Submit a guess and return similarity feedback.

        Despite older docs mentioning POST bodies, the live server uses GET.
        """
        guess_clean = guess.strip().lower().replace(" ", "_")
        secret_clean = secret_word.strip().lower().replace(" ", "_")
        path = f"/similarity/{guess_clean}/{secret_clean}/{self.language}"
        response = self._request("GET", path)
        try:
            payload = response.json()
        except ValueError as exc:
            raise SemantleAPIError("similarity response is not valid JSON") from exc
        try:
            return GuessResult(
                guess=guess_clean,
                similarity=float(payload["similarity"]),
                initial_similarity=float(payload["initialSimilarity"]),
                percentile=payload.get("percentile"),
                closest_similarity=float(payload["closestSimilarity"]),
                raw=payload,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SemantleAPIError(f"unexpected similarity payload: {payload!r}") from exc
