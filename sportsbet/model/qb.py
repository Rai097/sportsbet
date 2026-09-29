"""Quarterback-aware ratings: who starts at QB moves the line more than anything else.

Design (FiveThirtyEight-style, rebuilt on free nflverse data):
  * Per-game QB value = (passing EPA + rushing EPA) per play, scaled to a standard
    38-play start so it reads in points per game. Plays = attempts + sacks + carries.
    Adjusted for the opponent's rolling pass defence.
  * Each QB carries a decayed, play-weighted average of his game values, shrunk toward
    a replacement-level prior worth six starts. New or rarely used QBs therefore sit near
    replacement, which is what most backups are.
  * Team strength = plain Elo + points_per_value * (value(today's starter) - the team's
    rolling value of recent starters). A healthy starter adds ~0 (Elo already knows him);
    a backup costs the value gap. The listed starter comes from the schedule's
    home_qb_id / home_qb_name, else the team's last starter.

Fit (hyperparameters chosen on 2011-2018 with 2010 as burn-in, then frozen):
  * Grid over decay, prior weight, replacement level, team EMA, points per value,
    opponent adjustment and baseline. The surface is flat near the optimum (Brier within
    0.0003 across neighbours). Chosen: decay 0.93, prior 6 starts, replacement -8.0,
    team EMA 0.1, 0.8 spread points per value point, opponent adjustment on, offseason
    carry 0.5. A QB-neutral Elo ("league" baseline) was worse on the same first grid:
    best 0.2127 vs 0.2118 for the team baseline.
  * Replacement -8.0 is a shrinkage target, not an observed mean: raw early-career
    starts average -1.8, but those QBs were selected to start; unknown emergency starters
    are worse, and the lower prior scored better.
  * Results (Brier / log loss; regular season with a closing moneyline):
      2011-2018 (tuning)  plain 0.2166 / 0.6231  QB 0.2115 / 0.6113  market 0.2097 / 0.6081
      2015-2026           plain 0.2230 / 0.6388  QB 0.2192 / 0.6290  market 0.2120 / 0.6136
      2019-2026 (holdout) plain 0.2243 / 0.6421  QB 0.2207 / 0.6326  market 0.2106 / 0.6101
    Holdout Brier gain 0.0036 (paired SE 0.0015, t=2.4); spread MAE vs the closing line
    falls from 2.65 to 2.33 points. It still trails the market, and neither model makes
    money betting against the close (see backtest).

Market price of a QB change (measure_market_qb_price, within-season changes only,
residual = closing spread_line minus the plain Elo spread):
  * Regression of the residual on the change in QB value (home minus away), 706 games
    with a change, 2011-2026: slope 0.455 points per value point (0.395 on 2011-2018,
    0.504 on 2019-2026), R^2 0.24. The slope is attenuated by noise in the value change.
  * Established starter (started each of the last 4) replaced by a QB with fewer career
    starts who started none of them, opponent QB unchanged: 214 games, the market moved
    3.76 points against the team (median 3.65; 3.21 on 2011-2018, 4.23 on 2019-2026) for
    a mean value gap of 4.21, i.e. 0.89 points per value point, in line with the 0.8
    that best fits results. injuries.STARTER_POINTS["QB"] = 3.8 and
    injuries.QB_POINTS_PER_VALUE = 0.9 come from these numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sportsbet.model.elo import EloConfig, EloModel

PLAYS_PER_GAME = 38.0  # median QB attempts + sacks + carries in a full start


@dataclass
class QbConfig:
    """Defaults are the 2011-2018 Brier-optimal values (see module docstring)."""

    decay: float = 0.93  # weight kept by older starts each time a QB plays a full game
    prior_games: float = 6.0  # pseudo-starts at replacement level in every QB's average
    replacement: float = -8.0  # points per game (EPA scale) assumed for an unknown QB
    team_alpha: float = 0.1  # EMA step for the team's rolling starter value
    points_per_value: float = 0.8  # spread points per point of QB value gap
    season_carry: float = 0.5  # fraction of a QB's evidence kept across an offseason
    opp_adjust: bool = True  # subtract the opponent's rolling pass defence
    opp_alpha: float = 0.1
    baseline: str = "team"  # "team" (Elo includes the QB) or "league" (QB-neutral Elo)


def qb_game_values(player_stats: pd.DataFrame, schedule: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per QB per game: game_id, team, qb_id, qb_name, value, weight.

    value is EPA per play times PLAYS_PER_GAME; weight is the share of a full start he
    played (capped at 1), so a two-snap cameo barely moves his rating.
    """
    ps = player_stats
    starters: set[str] = set()
    if schedule is not None:
        starters = set(schedule["home_qb_id"].dropna()) | set(schedule["away_qb_id"].dropna())
    ps = ps[(ps["position"] == "QB") | ps["player_id"].isin(starters)]
    plays = ps["attempts"].fillna(0) + ps["sacks_suffered"].fillna(0) + ps["carries"].fillna(0)
    epa = ps["passing_epa"].fillna(0) + ps["rushing_epa"].fillna(0)
    out = pd.DataFrame(
        {
            "game_id": ps["game_id"],
            "team": ps["team"],
            "qb_id": ps["player_id"],
            "qb_name": ps.get("player_display_name", ps.get("player_name")),
            "value": np.where(plays > 0, epa / plays.where(plays > 0, 1) * PLAYS_PER_GAME, 0.0),
            "weight": (plays / PLAYS_PER_GAME).clip(upper=1.0),
        }
    )
    return out[plays > 0].reset_index(drop=True)


def load_qb_game_values(seasons: list[int], schedule: pd.DataFrame | None = None) -> pd.DataFrame:
    """Weekly nflverse player stats reduced to QB game values."""
    import nflreadpy as nfl  # lazy: slow import and network on first use

    ps = nfl.load_player_stats(seasons).to_pandas()
    return qb_game_values(ps, schedule)


@dataclass
class _QbState:
    sum_wv: float = 0.0
    sum_w: float = 0.0
    starts: int = 0


@dataclass
class QbRatings:
    """Rolling QB values, team baselines and last-known starters."""

    config: QbConfig = field(default_factory=QbConfig)
    qbs: dict[str, _QbState] = field(default_factory=dict)
    names: dict[str, str] = field(default_factory=dict)
    team_base: dict[str, float] = field(default_factory=dict)
    last_starter: dict[str, str] = field(default_factory=dict)
    def_allowed: dict[str, float] = field(default_factory=dict)
    league_mean: float = 0.0

    def value(self, qb_id: str | None) -> float:
        """Points-per-game value; unknown QBs are replacement level."""
        c = self.config
        s = self.qbs.get(qb_id) if qb_id else None
        if s is None:
            return c.replacement
        return (s.sum_wv + c.replacement * c.prior_games) / (s.sum_w + c.prior_games)

    def starts(self, qb_id: str | None) -> int:
        s = self.qbs.get(qb_id) if qb_id else None
        return s.starts if s else 0

    def above_replacement(self, qb_id: str | None) -> float:
        """Spread points this QB is worth over a replacement-level QB."""
        return self.config.points_per_value * (self.value(qb_id) - self.config.replacement)

    def adjustment(self, team: str, qb_id: str | None) -> float:
        """Spread points to add to a team's Elo line for this starter."""
        c = self.config
        if qb_id is None:
            qb_id = self.last_starter.get(team)
        if c.baseline == "league":
            return c.points_per_value * (self.value(qb_id) - self.league_mean)
        base = self.team_base.get(team)
        if base is None:
            return 0.0
        return c.points_per_value * (self.value(qb_id) - base)

    def new_season(self) -> None:
        carry = self.config.season_carry
        if carry >= 1.0:
            return
        for s in self.qbs.values():
            s.sum_wv *= carry
            s.sum_w *= carry

    def observe(self, team: str, opponent: str, starter: str | None, games: list[tuple[str, str, float, float]]) -> None:
        """Update after a game. games holds (qb_id, qb_name, value, weight) for this team's QBs."""
        c = self.config
        opp_adj = 0.0
        if c.opp_adjust and opponent in self.def_allowed:
            opp_adj = self.def_allowed[opponent] - self.league_mean
        team_raw = 0.0
        team_w = 0.0
        for qb_id, name, value, weight in games:
            s = self.qbs.setdefault(qb_id, _QbState())
            v = value - opp_adj
            s.sum_wv = s.sum_wv * c.decay ** weight + weight * v
            s.sum_w = s.sum_w * c.decay ** weight + weight
            self.names[qb_id] = name
            team_raw += weight * value
            team_w += weight
        if starter is not None:
            if starter in self.qbs:
                self.qbs[starter].starts += 1
            self.last_starter[team] = starter
            cur = self.value(starter)
            prev = self.team_base.get(team)
            self.team_base[team] = cur if prev is None else prev + c.team_alpha * (cur - prev)
        if team_w > 0:
            # Defence is rated on what the offence's QBs did against it, before adjustment.
            raw = team_raw / team_w
            d = self.def_allowed.get(opponent)
            self.def_allowed[opponent] = raw if d is None else d + c.opp_alpha * (raw - d)
            self.league_mean += 0.01 * (raw - self.league_mean)

    def id_for_name(self, team: str, name: str | None) -> str | None:
        """Best-effort name -> id, preferring the team's last starter on ties."""
        if not name or (isinstance(name, float) and math.isnan(name)):
            return None
        ids = [i for i, n in self.names.items() if n == name]
        if not ids:
            return None
        if self.last_starter.get(team) in ids:
            return self.last_starter[team]
        return max(ids, key=lambda i: self.qbs[i].sum_w)


def _nan_to_none(x: Any) -> Any:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) or x is pd.NA else x


@dataclass
class QbEloModel(EloModel):
    """Elo plus the starting-QB adjustment. Plain Elo behaviour is untouched."""

    qb: QbRatings = field(default_factory=QbRatings)
    game_values: dict[tuple[str, str], list[tuple[str, str, float, float]]] = field(default_factory=dict)

    @classmethod
    def from_game_values(cls, gv: pd.DataFrame, qb_config: QbConfig | None = None, elo_config: EloConfig | None = None) -> "QbEloModel":
        idx: dict[tuple[str, str], list[tuple[str, str, float, float]]] = {}
        for r in gv.itertuples(index=False):
            idx.setdefault((r.game_id, r.team), []).append((r.qb_id, r.qb_name, float(r.value), float(r.weight)))
        return cls(config=elo_config or EloConfig(), qb=QbRatings(config=qb_config or QbConfig()), game_values=idx)

    def _new_season(self, season: int) -> None:
        if self.last_season is not None and season != self.last_season:
            self.qb.new_season()
        super()._new_season(season)

    def starters(self, g: Any) -> tuple[str | None, str | None]:
        """Listed starters for a schedule row, by id, then by name, else last starter."""
        out = []
        for side in ("home", "away"):
            team = getattr(g, f"{side}_team")
            qid = _nan_to_none(getattr(g, f"{side}_qb_id", None))
            if qid is None:
                qid = self.qb.id_for_name(team, _nan_to_none(getattr(g, f"{side}_qb_name", None)))
            out.append(qid if qid is not None else self.qb.last_starter.get(team))
        return out[0], out[1]

    def qb_points(self, g: Any) -> float:
        """Spread-scale QB adjustment, positive favours home."""
        h, a = self.starters(g)
        return self.qb.adjustment(g.home_team, h) - self.qb.adjustment(g.away_team, a)

    def predict_game(self, g: Any, extra_points: float = 0.0) -> dict[str, float]:
        return super().predict_game(g, extra_points + self.qb_points(g))

    def update_game(self, g: Any) -> None:
        h, a = self.starters(g)
        neutral = getattr(g, "location", "Home") == "Neutral"
        # With a league baseline the Elo must learn QB-neutral strength, otherwise the QB
        # is counted twice. With a team baseline the plain update is what "Elo includes
        # the usual starter" assumes.
        extra = self.qb_points(g) if self.qb.config.baseline == "league" else 0.0
        self.update(g.home_team, g.away_team, float(g.home_score), float(g.away_score), neutral=neutral, extra_points=extra)
        for side, team, opp, starter in (("home", g.home_team, g.away_team, h), ("away", g.away_team, g.home_team, a)):
            name = _nan_to_none(getattr(g, f"{side}_qb_name", None))
            if starter is not None and name and starter not in self.qb.names:
                self.qb.names[starter] = name  # starters with no stats yet still get a name
            self.qb.observe(team, opp, starter, self.game_values.get((g.game_id, team), []))

    def qb_table(self) -> pd.DataFrame:
        """Current value of each team's last starter."""
        rows = []
        for team, qid in self.qb.last_starter.items():
            rows.append(
                {
                    "team": team,
                    "qb": self.qb.names.get(qid, qid),
                    "value": round(self.qb.value(qid), 2),
                    "pts_above_repl": round(self.qb.above_replacement(qid), 2),
                }
            )
        return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)


def qb_injury_gaps(model: QbEloModel, depth: pd.DataFrame) -> dict[tuple[str, str], float]:
    """(team, QB1 name) -> value lost if he sits and QB2 starts, from the latest depth chart.

    Feed this to injuries.team_impacts(qb_gaps=...). Handles the current nflverse depth
    chart schema (dt, team, player_name, gsis_id, pos_abb, pos_rank) and the legacy one
    (season, week, club_code, full_name, gsis_id, position, depth_team).
    """
    d = depth
    if "dt" in d.columns and not d.empty:
        d = d[d["dt"] == d["dt"].max()]
    elif "week" in d.columns and not d.empty:
        d = d[(d["season"] == d["season"].max())]
        d = d[d["week"] == d["week"].max()]
    pos_col = "pos_abb" if "pos_abb" in d.columns else "position"
    team_col = "club_code" if "club_code" in d.columns else "team"
    name_col = "full_name" if "full_name" in d.columns else "player_name"
    rank_col = "depth_team" if "depth_team" in d.columns else "pos_rank"
    q = d[d[pos_col] == "QB"].assign(_rank=pd.to_numeric(d[rank_col], errors="coerce"))
    out: dict[tuple[str, str], float] = {}
    for team, grp in q.sort_values("_rank").groupby(team_col):
        grp = grp.drop_duplicates(name_col)
        if len(grp) < 2:
            continue
        ids = []
        for r in grp.head(2).itertuples(index=False):
            qid = _nan_to_none(getattr(r, "gsis_id", None)) or model.qb.id_for_name(team, getattr(r, name_col))
            ids.append(qid)
        out[(team, grp.iloc[0][name_col])] = model.qb.value(ids[0]) - model.qb.value(ids[1])
    return out


@dataclass
class MarketQbPrice:
    """How the closing spread moves, beyond plain Elo, when a team changes QB."""

    n_changes: int  # games where at least one side's starter differs from its previous game
    slope: float  # market points per point of QB value change (home minus away)
    intercept: float
    r2: float
    n_backup: int  # established starter replaced by a QB who started none of the last few
    backup_shift: float  # mean market move against the team in those games, in points
    backup_value_gap: float  # mean QB value lost in those games (model units)
    backup_rows: pd.DataFrame | None = None

    @property
    def backup_points_per_value(self) -> float:
        """Market points per unit of value gap in starter -> backup games (ratio of means).

        Unlike the regression slope this is not attenuated by noise in the value change.
        """
        return self.backup_shift / self.backup_value_gap if self.backup_value_gap else float("nan")


def measure_market_qb_price(
    schedule: pd.DataFrame,
    game_values: pd.DataFrame,
    config: QbConfig | None = None,
    seasons: tuple[int, int] | None = None,
    established: int = 4,
) -> MarketQbPrice:
    """Regress (spread_line - plain Elo spread) on the pregame change in QB value.

    Both models walk forward together so every value is known before kickoff. Only
    within-season changes count; offseason changes mix in roster turnover.
    """
    plain = EloModel()
    qbm = QbEloModel.from_game_values(game_values, config)
    played = schedule[schedule["result"].notna()].sort_values(["season", "gameday", "gametime"])
    recent: dict[str, list[str | None]] = {}
    rows = []
    for g in played.itertuples(index=False):
        season = int(g.season)
        if plain.last_season != season:
            recent = {}
        plain._new_season(season)
        qbm._new_season(season)
        spread = plain.predict_game(g)["home_spread"]
        starters = qbm.starters(g)
        row: dict[str, Any] = {"game_id": g.game_id, "season": season, "spread_line": g.spread_line, "elo_spread": spread}
        for side, team, qid in (("home", g.home_team, starters[0]), ("away", g.away_team, starters[1])):
            hist = recent.get(team, [])
            prev = hist[-1] if hist else None
            changed = prev is not None and qid is not None and qid != prev
            row[f"{side}_delta"] = qbm.qb.value(qid) - qbm.qb.value(prev) if changed else 0.0
            row[f"{side}_changed"] = changed
            # A backup: the incumbent started every recent game, the new man none of them,
            # and he has fewer career starts (so a veteran returning from injury is not one).
            row[f"{side}_to_backup"] = (
                changed
                and len(hist) >= established
                and all(h == prev for h in hist[-established:])
                and qid not in hist[-established:]
                and qbm.qb.starts(qid) < qbm.qb.starts(prev)
            )
            row[f"{side}_qb"] = qbm.qb.names.get(qid, qid) if qid else None
            row[f"{side}_prev_qb"] = qbm.qb.names.get(prev, prev) if prev else None
            recent[team] = (hist + [qid])[-established:]
        rows.append(row)
        plain.update_game(g)
        qbm.update_game(g)

    df = pd.DataFrame(rows)
    df = df[df["spread_line"].notna()]
    if seasons is not None:
        df = df[(df["season"] >= seasons[0]) & (df["season"] <= seasons[1])]
    df["resid"] = df["spread_line"] - df["elo_spread"]
    df["x"] = df["home_delta"] - df["away_delta"]

    chg = df[df["home_changed"] | df["away_changed"]]
    if len(chg) >= 3 and chg["x"].var() > 0:
        slope, intercept = np.polyfit(chg["x"], chg["resid"], 1)
        pred = intercept + slope * chg["x"]
        r2 = 1.0 - float(((chg["resid"] - pred) ** 2).sum() / ((chg["resid"] - chg["resid"].mean()) ** 2).sum())
    else:
        slope, intercept, r2 = float("nan"), float("nan"), float("nan")

    # Team-perspective move for starter -> backup, net of the same side's usual residual
    # in games where neither QB changed (plain Elo has its own home/away bias).
    calm = df[~df["home_changed"] & ~df["away_changed"]]
    shifts: list[float] = []
    gaps: list[float] = []
    ev_rows = []
    for side, sign, other in (("home", 1.0, "away"), ("away", -1.0, "home")):
        base = sign * calm["resid"].mean() if len(calm) else 0.0
        ev = df[df[f"{side}_to_backup"] & ~df[f"{other}_changed"]]
        shifts += list(-(sign * ev["resid"] - base))
        gaps += list(-ev[f"{side}_delta"])
        ev_rows.append(ev.assign(side=side))
    return MarketQbPrice(
        n_changes=int(len(chg)),
        slope=float(slope),
        intercept=float(intercept),
        r2=float(r2),
        n_backup=len(shifts),
        backup_shift=float(np.mean(shifts)) if shifts else float("nan"),
        backup_value_gap=float(np.mean(gaps)) if gaps else float("nan"),
        backup_rows=pd.concat(ev_rows, ignore_index=True),
    )
