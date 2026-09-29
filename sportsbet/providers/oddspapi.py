"""Second odds source: OddsPapi (https://oddspapi.io), free tier 250 requests a month.

The Odds API stopped carrying Caesars for NFL, and Caesars is one of the two books the
user can legally bet. OddsPapi documents Caesars coverage. This module holds the thin
client and the discovery calls; the normaliser is written against the real payload shape
once a probe from GitHub Actions has printed it (the dev container cannot reach the host).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import requests

log = logging.getLogger(__name__)

BASE = "https://api.oddspapi.io/v4"


@dataclass
class OddsPapiClient:
    api_key: str
    base: str = BASE
    session: requests.Session = field(default_factory=requests.Session)
    calls: int = 0

    def get(self, path: str, **params: Any) -> Any:
        params = {**params, "apiKey": self.api_key}
        r = self.session.get(f"{self.base}{path}", params=params, timeout=30)
        self.calls += 1
        if r.status_code >= 400:
            body = r.text[:300].replace(self.api_key, "<KEY>")
            raise RuntimeError(f"oddspapi {path} -> {r.status_code}: {body}")
        return r.json()

    def bookmakers(self) -> Any:
        return self.get("/bookmakers")

    def sports(self) -> Any:
        return self.get("/sports")

    def tournaments(self, **params: Any) -> Any:
        return self.get("/tournaments", **params)

    def odds_by_tournaments(self, bookmaker: str, tournament_ids: str, **params: Any) -> Any:
        return self.get("/odds-by-tournaments", bookmaker=bookmaker, tournamentIds=tournament_ids, **params)


def _find(items: Any, *needles: str, keys: tuple[str, ...] = ("name", "slug", "id", "bookmaker", "title")) -> list[Any]:
    """Entries whose text fields mention any needle (case-insensitive). Tolerant of shape."""
    if isinstance(items, dict):
        items = items.get("data") or items.get("items") or list(items.values())
    out = []
    for it in items or []:
        text = json.dumps(it).lower() if not isinstance(it, str) else it.lower()
        if any(n in text for n in needles):
            out.append(it)
    return out


def discover(client: OddsPapiClient) -> dict[str, Any]:
    """Three cheap calls: Caesars bookmaker entry, NFL sport/tournament ids, one odds sample."""
    found: dict[str, Any] = {}
    books = client.bookmakers()
    found["caesars_bookmakers"] = _find(books, "caesars", "william")
    found["bookmaker_sample"] = (books if isinstance(books, list) else list(books.values()) if isinstance(books, dict) else [books])[:2]
    sports = client.sports()
    found["football_sports"] = _find(sports, "american football", "americanfootball", "nfl")
    tours = client.tournaments()
    found["nfl_tournaments"] = _find(tours, "nfl", "national football league")
    return found
