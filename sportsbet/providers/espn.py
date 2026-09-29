"""ESPN's public (undocumented) NFL injuries feed.

Official league injury reports only publish Wednesday through Friday. ESPN updates
its feed as news breaks, which is what makes it useful for catching a line that has
not yet moved. The endpoint is unofficial, so parsing is deliberately tolerant.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests

from sportsbet.teams import to_abbr

log = logging.getLogger(__name__)

# ESPN statuses normalised to the league's vocabulary.
STATUS_MAP = {
    "out": "Out",
    "injured reserve": "Out",
    "ir": "Out",
    "doubtful": "Doubtful",
    "questionable": "Questionable",
    "day-to-day": "Questionable",
    "probable": "Probable",
    "active": "Active",
}


@dataclass
class InjuryRow:
    fetched_at: datetime
    source: str
    team: str
    player: str
    position: str | None
    status: str
    detail: str | None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def normalize_status(raw: str | None) -> str:
    if not raw:
        return "Unknown"
    return STATUS_MAP.get(raw.strip().lower(), raw.strip().title())


def parse_espn_injuries(payload: dict[str, Any], fetched_at: datetime | None = None) -> list[InjuryRow]:
    fetched_at = fetched_at or datetime.now(timezone.utc)
    rows: list[InjuryRow] = []
    for team_block in payload.get("injuries", []):
        team_name = team_block.get("displayName") or team_block.get("team", {}).get("displayName")
        try:
            team = to_abbr(team_name)
        except KeyError:
            log.warning("Unknown ESPN team %r", team_name)
            continue
        for inj in team_block.get("injuries", []):
            athlete = inj.get("athlete", {})
            pos = athlete.get("position", {})
            details = inj.get("details", {}) or {}
            rows.append(
                InjuryRow(
                    fetched_at=fetched_at,
                    source="espn",
                    team=team,
                    player=athlete.get("displayName", "?"),
                    position=pos.get("abbreviation") if isinstance(pos, dict) else None,
                    status=normalize_status(inj.get("status")),
                    detail=details.get("type") or inj.get("shortComment") or inj.get("longComment"),
                )
            )
    return rows


# ESPN's edge returns 403 to non-browser user agents, so look like a browser.
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.espn.com/nfl/injuries",
    "Origin": "https://www.espn.com",
}


def fetch_espn_injuries(url: str, session: requests.Session | None = None) -> list[InjuryRow]:
    session = session or requests.Session()
    resp = session.get(url, timeout=30, headers=BROWSER_HEADERS)
    if resp.status_code >= 400:
        # include a slice of the body so an unattended log shows what ESPN actually said
        snippet = resp.text[:200].replace("\n", " ")
        raise requests.HTTPError(
            f"{resp.status_code} from ESPN injuries feed: {snippet!r}", response=resp
        )
    return parse_espn_injuries(resp.json())
