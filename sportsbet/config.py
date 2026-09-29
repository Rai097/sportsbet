"""Runtime configuration. Everything is overridable via environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("SPORTSBET_DATA_DIR", PROJECT_ROOT / "data"))

# The Odds API bookmaker keys. Caesars is listed under its legacy William Hill key.
TARGET_BOOKS = {"betmgm": "BetMGM", "williamhill_us": "Caesars"}

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
    espn_injuries_url: str = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
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
