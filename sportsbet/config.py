"""Runtime configuration. Everything is overridable via environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader so a local key works without exporting it. Never overrides the shell."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv(PROJECT_ROOT / ".env")
DATA_DIR = Path(os.environ.get("SPORTSBET_DATA_DIR", PROJECT_ROOT / "data"))

# The Odds API bookmaker keys. Caesars is listed under its legacy William Hill key.
# A live probe on 2026-09-29 found no Caesars feed for NFL at all; the key stays so it
# is picked up if it returns. Override with SPORTSBET_TARGET_BOOKS="betmgm,fanduel,...".
_KNOWN_BOOKS = {
    "betmgm": "BetMGM",
    "williamhill_us": "Caesars",
    "caesars": "Caesars",
    "draftkings": "DraftKings",
    "fanduel": "FanDuel",
    "betrivers": "BetRivers",
    "hardrockbet": "Hard Rock Bet",
    "ballybet": "Bally Bet",
    "espnbet": "theScore Bet",
    "fliff": "Fliff",
    "betparx": "betPARX",
}
_target_keys = [k.strip() for k in os.environ.get("SPORTSBET_TARGET_BOOKS", "betmgm,williamhill_us").split(",") if k.strip()]
TARGET_BOOKS = {k: _KNOWN_BOOKS.get(k, k) for k in _target_keys}

# Books used to estimate the fair (no-vig) price. Pinnacle is the sharpest public
# reference. The rest form a consensus fallback when Pinnacle has no line.
REFERENCE_BOOKS = ["pinnacle", "draftkings", "fanduel", "betrivers", "bovada", "betonlineag"]
SHARP_BOOK = "pinnacle"

MARKETS = ["h2h", "spreads", "totals"]
SPORT_KEY = "americanfootball_nfl"


@dataclass
class Settings:
    odds_api_key: str | None = field(default_factory=lambda: os.environ.get("ODDS_API_KEY"))
    odds_api_base: str = "https://api.the-odds-api.com/v4"
    # site.api.espn.com answers 403 to datacenter IPs (GitHub runners); site.web.api does not.
    espn_injuries_url: str = "https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
    data_dir: Path = DATA_DIR
    db_path: Path = field(default_factory=lambda: DATA_DIR / "sportsbet.duckdb")
    min_ev: float = float(os.environ.get("SPORTSBET_MIN_EV", "0.01"))
    kelly_fraction: float = float(os.environ.get("SPORTSBET_KELLY_FRACTION", "0.25"))
    # Free tier of The Odds API is 500 credits per month. Refuse to spend below this floor.
    quota_floor: int = int(os.environ.get("SPORTSBET_QUOTA_FLOOR", "25"))

    def __post_init__(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    return Settings()
