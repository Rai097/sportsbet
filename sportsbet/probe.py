"""Live connectivity probe for unattended runs.

The development container cannot reach the odds or ESPN hosts, so this command exists to
be run from GitHub Actions (workflow input `probe`) and answer two questions in its log:
which bookmaker keys the odds feed really returns for NFL right now, and which ESPN
endpoint, if any, serves injuries to a datacenter IP. It spends 2 API credits.
"""

from __future__ import annotations

import sys

import requests

from sportsbet.config import SPORT_KEY, load_settings
from sportsbet.providers.espn import BROWSER_HEADERS

ESPN_CANDIDATES = [
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries",
    "https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/injuries",
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/2/injuries",
    "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams/2/injuries",
    "https://cdn.espn.com/core/nfl/injuries?xhr=1",
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",  # control
]


def probe_espn(session: requests.Session | None = None) -> list[tuple[str, str]]:
    session = session or requests.Session()
    out = []
    for url in ESPN_CANDIDATES:
        try:
            r = session.get(url, timeout=20, headers=BROWSER_HEADERS)
            body = r.text[:100].replace("\n", " ")
            out.append((url, f"{r.status_code} {r.headers.get('content-type', '')[:30]} {body!r}"))
        except requests.RequestException as exc:
            out.append((url, f"error {type(exc).__name__}: {str(exc)[:120]}"))
    return out


def probe_odds_books(settings, session: requests.Session | None = None) -> tuple[dict[str, str], dict]:
    """Every bookmaker key/title present for NFL across the us and us2 regions (2 credits)."""
    session = session or requests.Session()
    r = session.get(
        f"{settings.odds_api_base}/sports/{SPORT_KEY}/odds",
        params={"apiKey": settings.odds_api_key, "regions": "us,us2", "markets": "h2h", "oddsFormat": "american"},
        timeout=30,
    )
    quota = {k: r.headers.get(k) for k in ("x-requests-remaining", "x-requests-used", "x-requests-last")}
    if r.status_code >= 400:
        raise RuntimeError(f"odds api {r.status_code}: {r.text[:200].replace(settings.odds_api_key, '<KEY>')}")
    books: dict[str, str] = {}
    for ev in r.json():
        for b in ev.get("bookmakers", []):
            books[b["key"]] = b.get("title", "")
    return books, quota


def cmd_probe(args) -> int:
    """Print which bookmakers the odds feed returns for NFL and which ESPN endpoints answer."""
    settings = load_settings()
    print("== ESPN endpoints")
    for url, result in probe_espn():
        print(f"{result}\n    {url}")
    print("== Odds API bookmakers for NFL (us + us2)")
    if not settings.odds_api_key:
        print("ODDS_API_KEY not set; skipping")
        return 0
    try:
        books, quota = probe_odds_books(settings)
    except (requests.RequestException, RuntimeError) as exc:
        print(f"odds probe failed: {str(exc).replace(settings.odds_api_key, '<KEY>')}", file=sys.stderr)
        return 0
    for key, title in sorted(books.items()):
        print(f"{key:<20} {title}")
    print(f"quota: {quota}")
    return 0


def register(subparsers) -> None:
    s = subparsers.add_parser("probe", help=cmd_probe.__doc__)
    s.set_defaults(func=cmd_probe)
