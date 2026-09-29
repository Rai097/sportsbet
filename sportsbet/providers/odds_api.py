"""Client for The Odds API v4 (https://the-odds-api.com).

Free tier: 500 credits per month. A request costs (markets x regions); when the
`bookmakers` parameter is used instead of `regions`, every block of up to 10
bookmakers is billed as one region. Asking for 3 markets across up to 10 named
books therefore costs 3 credits, which allows roughly 5 pulls a day for a month.
The client reads the quota headers on every response and refuses to spend below a
configured floor so a busy Sunday never burns the whole month.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from sportsbet.config import MARKETS, REFERENCE_BOOKS, SPORT_KEY, TARGET_BOOKS, Settings
from sportsbet.teams import to_abbr

log = logging.getLogger(__name__)


class QuotaExhausted(RuntimeError):
    pass


@dataclass
class Quota:
    remaining: int | None = None
    used: int | None = None
    last_cost: int | None = None

    def update_from_headers(self, headers: Any) -> None:
        def _int(key: str) -> int | None:
            v = headers.get(key)
            try:
                return int(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        self.remaining = _int("x-requests-remaining")
        self.used = _int("x-requests-used")
        self.last_cost = _int("x-requests-last")


@dataclass
class OddsRow:
    fetched_at: datetime
    event_id: str
    commence_time: datetime
    home_team: str  # nflverse abbreviation
    away_team: str
    bookmaker: str
    market: str  # h2h | spreads | totals
    outcome: str  # team abbreviation, or Over / Under
    point: float | None
    price: float  # American odds
    last_update: datetime | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "fetched_at": self.fetched_at,
            "event_id": self.event_id,
            "commence_time": self.commence_time,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "bookmaker": self.bookmaker,
            "market": self.market,
            "outcome": self.outcome,
            "point": self.point,
            "price": self.price,
            "last_update": self.last_update,
        }


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def normalize(raw: list[dict[str, Any]], fetched_at: datetime | None = None) -> list[OddsRow]:
    """Flatten the nested Odds API payload into one row per (book, market, outcome)."""
    fetched_at = fetched_at or datetime.now(timezone.utc)
    rows: list[OddsRow] = []
    for event in raw:
        try:
            home = to_abbr(event["home_team"])
            away = to_abbr(event["away_team"])
        except KeyError as exc:
            log.warning("Skipping event with unknown team: %s", exc)
            continue
        commence = _parse_ts(event.get("commence_time"))
        for book in event.get("bookmakers", []):
            book_key = book["key"]
            book_updated = _parse_ts(book.get("last_update"))
            for market in book.get("markets", []):
                mkey = market["key"]
                for outcome in market.get("outcomes", []):
                    name = outcome["name"]
                    if mkey in ("h2h", "spreads"):
                        try:
                            name = to_abbr(name)
                        except KeyError:
                            log.warning("Unknown outcome team %r in %s", name, mkey)
                            continue
                    rows.append(
                        OddsRow(
                            fetched_at=fetched_at,
                            event_id=event["id"],
                            commence_time=commence,
                            home_team=home,
                            away_team=away,
                            bookmaker=book_key,
                            market=mkey,
                            outcome=name,
                            point=outcome.get("point"),
                            price=float(outcome["price"]),
                            last_update=_parse_ts(market.get("last_update")) or book_updated,
                        )
                    )
    return rows


@dataclass
class OddsApiClient:
    settings: Settings
    session: requests.Session = field(default_factory=requests.Session)
    quota: Quota = field(default_factory=Quota)

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        if not self.settings.odds_api_key:
            raise RuntimeError("ODDS_API_KEY is not set. Get a free key at https://the-odds-api.com")
        if self.quota.remaining is not None and self.quota.remaining < self.settings.quota_floor:
            raise QuotaExhausted(
                f"Only {self.quota.remaining} credits left, below floor {self.settings.quota_floor}"
            )
        key = self.settings.odds_api_key
        params = {**params, "apiKey": key}
        try:
            resp = self.session.get(f"{self.settings.odds_api_base}{path}", params=params, timeout=30)
            self.quota.update_from_headers(resp.headers)
            log.info(
                "odds api %s cost=%s remaining=%s", path, self.quota.last_cost, self.quota.remaining
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            # requests puts the full URL, key included, in its messages; callers print them
            msg = str(exc).replace(key, "<ODDS_API_KEY>")
            raise type(exc)(msg, response=exc.response) from None
        return resp.json()

    def fetch_nfl_odds(
        self,
        markets: list[str] | None = None,
        bookmakers: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        markets = markets or MARKETS
        bookmakers = bookmakers or (list(TARGET_BOOKS) + REFERENCE_BOOKS)
        if len(bookmakers) > 10:
            log.warning("More than 10 bookmakers doubles the credit cost of this call")
        return self._get(
            f"/sports/{SPORT_KEY}/odds",
            {
                "markets": ",".join(markets),
                "bookmakers": ",".join(bookmakers),
                "oddsFormat": "american",
                "dateFormat": "iso",
            },
        )

    def fetch_and_normalize(self) -> list[OddsRow]:
        raw = self.fetch_nfl_odds()
        return normalize(raw)


def load_fixture(path: Path) -> list[dict[str, Any]]:
    with open(path) as fh:
        return json.load(fh)
