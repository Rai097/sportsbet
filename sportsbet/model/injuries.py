"""Turn an injury report into a point-spread adjustment.

Non-QB weights are a documented heuristic, not a fitted model. Each position carries a
rough "points of spread" value for a starter being unavailable; they are small and mostly
matter in aggregate. Depth charts decide who is a starter. Doubtful players count at 80%,
Questionable at 35% (roughly historical play rates).

The QB weight is measured from the market (sportsbet.model.qb.measure_market_qb_price):
across 214 games from 2011-2026 where an established starter gave way to a backup, the
closing spread moved 3.8 points against the team beyond what plain Elo expected
(3.2 on 2011-2018, 4.2 on 2019-2026). When the QB model knows both the starter's and the
next man's value, the market's price per unit of value gap (0.9) replaces the flat 3.8.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

STARTER_POINTS = {
    "QB": 3.8,  # market move for starter -> backup, see module docstring
    "RB": 0.6,
    "WR": 0.7,
    "TE": 0.4,
    "T": 0.6,
    "OT": 0.6,
    "G": 0.4,
    "OG": 0.4,
    "C": 0.4,
    "OL": 0.4,
    "DE": 0.5,
    "EDGE": 0.6,
    "OLB": 0.4,
    "DT": 0.4,
    "NT": 0.3,
    "DL": 0.4,
    "LB": 0.35,
    "ILB": 0.35,
    "MLB": 0.35,
    "CB": 0.5,
    "S": 0.35,
    "FS": 0.35,
    "SS": 0.35,
    "DB": 0.3,
    "K": 0.4,
    "P": 0.1,
}
BACKUP_FRACTION = 0.15  # a non-starter counts for this fraction of the starter value
STATUS_WEIGHT = {"Out": 1.0, "Doubtful": 0.8, "Questionable": 0.35}
TEAM_CAP = 10.0  # never move a line more than this on injuries alone
# Market points per unit of QB value gap (qb.py value units), from the same backup games:
# 3.76 points moved / 4.21 value lost. 2011-2018 alone gives 0.78; the outcome-fitted
# model weight is 0.8, so the market and results agree to within noise.
QB_POINTS_PER_VALUE = 0.9


@dataclass
class InjuryImpact:
    team: str
    points: float
    detail: list[str]


def starters_from_depth_chart(depth: pd.DataFrame, season: int | None = None, week: int | None = None) -> set[tuple[str, str]]:
    """(team, player_name) pairs ranked first at their slot on the most recent depth chart.

    Handles the current nflverse schema (dt, team, player_name, pos_rank) and the
    legacy one (season, week, club_code, full_name, depth_team).
    """
    d = depth
    if "season" in d.columns and season is not None:
        d = d[d["season"] == season]
        if week is not None and "week" in d.columns and d["week"].notna().any():
            avail = d[d["week"] <= week]
            if not avail.empty:
                d = avail[avail["week"] == avail["week"].max()]
    elif "dt" in d.columns and not d.empty:
        d = d[d["dt"] == d["dt"].max()]
    team_col = "club_code" if "club_code" in d.columns else "team"
    name_col = "full_name" if "full_name" in d.columns else "player_name"
    rank_col = "depth_team" if "depth_team" in d.columns else "pos_rank"
    ranks = pd.to_numeric(d[rank_col], errors="coerce")
    first = d[ranks == 1]
    return set(zip(first[team_col], first[name_col]))


def team_impacts(
    injuries: pd.DataFrame,
    starters: set[tuple[str, str]] | None,
    status_col: str = "status",
    player_col: str = "player",
    position_col: str = "position",
    qb_gaps: dict[tuple[str, str], float] | None = None,
) -> dict[str, InjuryImpact]:
    """Aggregate per-team point impact from an injury table.

    qb_gaps maps (team, starting QB) to his value over the next QB up (see
    qb.qb_injury_gaps). When present it prices that QB instead of the flat weight.
    """
    out: dict[str, InjuryImpact] = {}
    for row in injuries.itertuples(index=False):
        status = getattr(row, status_col)
        weight = STATUS_WEIGHT.get(status)
        if weight is None:
            continue
        team = getattr(row, "team")
        pos = (getattr(row, position_col) or "").upper()
        base = STARTER_POINTS.get(pos)
        if base is None:
            continue
        player = getattr(row, player_col)
        is_starter = starters is None or (team, player) in starters
        pts = base * weight * (1.0 if is_starter else BACKUP_FRACTION)
        if pos == "QB" and qb_gaps and (team, player) in qb_gaps:
            # A known value gap beats the flat weight; a backup as good as the starter costs ~0.
            is_starter = True
            pts = max(qb_gaps[(team, player)], 0.0) * QB_POINTS_PER_VALUE * weight
        if pts < 0.05:
            continue
        impact = out.setdefault(team, InjuryImpact(team=team, points=0.0, detail=[]))
        impact.points += pts
        impact.detail.append(f"{player} ({pos}, {status}{'' if is_starter else ', backup'}) -{pts:.1f}")
    for impact in out.values():
        impact.points = min(impact.points, TEAM_CAP)
    return out


def matchup_adjustment(impacts: dict[str, InjuryImpact], home: str, away: str) -> float:
    """Spread-scale adjustment favouring home when positive."""
    home_loss = impacts[home].points if home in impacts else 0.0
    away_loss = impacts[away].points if away in impacts else 0.0
    return away_loss - home_loss


def from_nflverse_report(report: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Reshape the official weekly report into the columns team_impacts expects."""
    r = report[(report["season"] == season) & (report["week"] == week)]
    return pd.DataFrame(
        {
            "team": r["team"],
            "player": r["full_name"],
            "position": r["position"],
            "status": r["report_status"].fillna(""),
        }
    )
