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


def fetch_espn_injuries(url: str, session: requests.Session | None = None) -> list[InjuryRow]:
    session = session or requests.Session()
    resp = session.get(url, timeout=30, headers={"User-Agent": "sportsbet/0.1"})
    resp.raise_for_status()
    return parse_espn_injuries(resp.json())
