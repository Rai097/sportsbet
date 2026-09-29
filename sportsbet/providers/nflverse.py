"""Free NFL data from the nflverse project via nflreadpy.

Schedules include closing spread, total and moneylines for every game since 2010,
which makes them a free backtest set and a free reference line for the current week.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from sportsbet.teams import ABBR_ALIASES

log = logging.getLogger(__name__)


def _nfl():
    import nflreadpy as nfl  # imported lazily: it is slow and needs network on first use

    return nfl


def _fix_abbrs(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].replace(ABBR_ALIASES)
    return df


def load_schedules(seasons: list[int]) -> pd.DataFrame:
    df = _nfl().load_schedules(seasons).to_pandas()
    df = _fix_abbrs(df, ["home_team", "away_team"])
    df["gameday"] = pd.to_datetime(df["gameday"])
    return df


def load_injuries(seasons: list[int]) -> pd.DataFrame:
    df = _nfl().load_injuries(seasons).to_pandas()
    return _fix_abbrs(df, ["team"])


def load_depth_charts(seasons: list[int]) -> pd.DataFrame:
    df = _nfl().load_depth_charts(seasons).to_pandas()
    return _fix_abbrs(df, ["team", "club_code"])


def current_season(today: date | None = None) -> int:
    today = today or date.today()
    # The NFL season is labelled by the year it starts in; games run into the next February.
    return today.year if today.month >= 3 else today.year - 1


def current_week(schedule: pd.DataFrame, today: date | None = None) -> int:
    """First regular-season week that still has an unplayed game on or after today."""
    today = pd.Timestamp(today or date.today())
    reg = schedule[schedule["game_type"] == "REG"]
    pending = reg[reg["result"].isna() & (reg["gameday"] >= today.normalize())]
    if pending.empty:
        return int(reg["week"].max())
    return int(pending["week"].min())


def week_games(schedule: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    mask = (schedule["season"] == season) & (schedule["week"] == week)
    cols = [
        "game_id",
        "gameday",
        "gametime",
        "away_team",
        "home_team",
        "spread_line",
        "total_line",
        "away_moneyline",
        "home_moneyline",
        "away_qb_name",
        "home_qb_name",
        "result",
    ]
    return schedule.loc[mask, cols].reset_index(drop=True)
