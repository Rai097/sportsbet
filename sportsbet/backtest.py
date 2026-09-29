"""Backtest the Elo models (plain and QB-aware) against nflverse closing lines.

Two questions are answered honestly:
  1. Is the model better calibrated than the closing moneyline? (Brier and log loss)
  2. Would betting model-vs-market disagreements have made money at the listed odds?
Historical prices from BetMGM and Caesars are not free, so this measures the model,
not the book-shopping edge. The market columns are the closing consensus line.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from sportsbet.model.elo import EloModel
from sportsbet.model.qb import QbConfig, QbEloModel
from sportsbet.pricing import american_to_decimal, fair_prob_from_two_way


@dataclass
class BacktestResult:
    games: pd.DataFrame
    metrics: dict[str, float] = field(default_factory=dict)
    bets: pd.DataFrame | None = None

    def summary(self) -> str:
        lines = [f"{k:<32} {v:>10.4f}" for k, v in self.metrics.items()]
        return "\n".join(lines)


def _brier(p: pd.Series, y: pd.Series) -> float:
    return float(((p - y) ** 2).mean())


def _logloss(p: pd.Series, y: pd.Series) -> float:
    eps = 1e-6
    p = p.clip(eps, 1 - eps)
    return float(-(y * p.map(math.log) + (1 - y) * (1 - p).map(math.log)).mean())


def walk_forward(schedule: pd.DataFrame, eval_from: int, model: EloModel | None = None) -> pd.DataFrame:
    """Predict each game before updating on it. Games before eval_from only train."""
    model = model or EloModel()
    played = schedule[schedule["result"].notna()].sort_values(["season", "gameday", "gametime"]).copy()
    preds = []
    for g in played.itertuples(index=False):
        model._new_season(int(g.season))
        pred = model.predict_game(g)
        if g.season >= eval_from:
            preds.append({"game_id": g.game_id, **pred})
        model.update_game(g)
    out = played.merge(pd.DataFrame(preds), on="game_id", how="inner")
    out["home_win"] = (out["result"] > 0).astype(float)
    out.loc[out["result"] == 0, "home_win"] = 0.5
    return out


def _strategy_bets(df: pd.DataFrame, prob_col: str, spread_col: str, ml_edge: float, ats_edge: float) -> list[dict]:
    """Flat one-unit bets wherever the model disagrees with the closing market."""
    # Moneyline strategy: bet the side where model prob exceeds market fair prob by ml_edge.
    home_edge = df[prob_col] - df["market_home_prob"]
    bets = []
    for r, edge in zip(df.itertuples(index=False), home_edge):
        if edge >= ml_edge:
            side, price, won = "home", r.home_moneyline, r.result > 0
        elif edge <= -ml_edge:
            side, price, won = "away", r.away_moneyline, r.result < 0
        else:
            continue
        dec = american_to_decimal(price)
        pnl = (dec - 1.0) if won else (0.0 if r.result == 0 else -1.0)
        bets.append({"game_id": r.game_id, "season": r.season, "week": r.week, "type": "ml", "side": side, "price": price, "pnl": pnl})

    # Spread strategy: nflverse spread_line is the home margin the market expects
    # (positive = home favoured). Bet when the model disagrees by ats_edge points at -110.
    diff = df[spread_col] - df["spread_line"]
    for r, d in zip(df.itertuples(index=False), diff):
        if abs(d) < ats_edge:
            continue
        cover_margin = r.result - r.spread_line  # >0 home covers
        if d > 0:
            pnl = 100 / 110 if cover_margin > 0 else 0.0 if cover_margin == 0 else -1.0
            side = "home"
        else:
            pnl = 100 / 110 if cover_margin < 0 else 0.0 if cover_margin == 0 else -1.0
            side = "away"
        bets.append({"game_id": r.game_id, "season": r.season, "week": r.week, "type": "ats", "side": side, "price": -110, "pnl": pnl})
    return bets


def _roi_metrics(bets_df: pd.DataFrame, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for t in ("ml", "ats"):
        sub = bets_df[bets_df["type"] == t] if not bets_df.empty else bets_df
        out[f"{prefix}{t}_bets"] = float(len(sub))
        out[f"{prefix}{t}_roi"] = float(sub["pnl"].mean()) if len(sub) else 0.0
        out[f"{prefix}{t}_win_rate"] = float((sub["pnl"] > 0).mean()) if len(sub) else 0.0
    return out


def evaluate(
    schedule: pd.DataFrame,
    eval_from: int = 2015,
    ml_edge: float = 0.03,
    ats_edge: float = 2.0,
    qb_values: pd.DataFrame | None = None,
    qb_config: QbConfig | None = None,
) -> BacktestResult:
    """Plain Elo vs the closing line; with qb_values, the QB-aware model too.

    Plain Elo metrics keep their original keys. QB-aware metrics are prefixed "qb_" (or
    suffixed "_qb" for Brier/log loss) and computed on exactly the same games. The bets
    frame carries a "model" column ("elo" or "qb").
    """
    df = walk_forward(schedule, eval_from)
    if qb_values is not None:
        qdf = walk_forward(schedule, eval_from, QbEloModel.from_game_values(qb_values, qb_config))
        qdf = qdf[["game_id", "home_win_prob", "home_spread"]].rename(
            columns={"home_win_prob": "qb_home_win_prob", "home_spread": "qb_home_spread"}
        )
        df = df.merge(qdf, on="game_id", how="left")
    df = df[df["home_moneyline"].notna() & df["away_moneyline"].notna() & (df["game_type"] == "REG")].copy()

    fair = df.apply(lambda r: fair_prob_from_two_way(r.home_moneyline, r.away_moneyline), axis=1)
    df["market_home_prob"] = [f[0] for f in fair]

    metrics: dict[str, float] = {
        "games": float(len(df)),
        "brier_model": _brier(df["home_win_prob"], df["home_win"]),
        "brier_market": _brier(df["market_home_prob"], df["home_win"]),
        "logloss_model": _logloss(df["home_win_prob"], df["home_win"]),
        "logloss_market": _logloss(df["market_home_prob"], df["home_win"]),
    }
    bets_df = pd.DataFrame(_strategy_bets(df, "home_win_prob", "home_spread", ml_edge, ats_edge))
    metrics.update(_roi_metrics(bets_df))
    if not bets_df.empty:
        bets_df["model"] = "elo"

    if qb_values is not None:
        metrics["brier_qb"] = _brier(df["qb_home_win_prob"], df["home_win"])
        metrics["logloss_qb"] = _logloss(df["qb_home_win_prob"], df["home_win"])
        qb_bets = pd.DataFrame(_strategy_bets(df, "qb_home_win_prob", "qb_home_spread", ml_edge, ats_edge))
        metrics.update(_roi_metrics(qb_bets, prefix="qb_"))
        if not qb_bets.empty:
            qb_bets["model"] = "qb"
            bets_df = pd.concat([bets_df, qb_bets], ignore_index=True)
    return BacktestResult(games=df, metrics=metrics, bets=bets_df)


def comparison_table(metrics: dict[str, float]) -> pd.DataFrame:
    """Side-by-side plain Elo / QB-aware / market view of evaluate() metrics."""
    cols = {"elo": ("brier_model", "logloss_model", "ml_roi", "ml_bets", "ats_roi", "ats_bets")}
    if "brier_qb" in metrics:
        cols["qb_elo"] = ("brier_qb", "logloss_qb", "qb_ml_roi", "qb_ml_bets", "qb_ats_roi", "qb_ats_bets")
    cols["market"] = ("brier_market", "logloss_market", None, None, None, None)
    index = ["brier", "logloss", "ml_roi", "ml_bets", "ats_roi", "ats_bets"]
    data = {name: [metrics.get(k) if k else None for k in keys] for name, keys in cols.items()}
    return pd.DataFrame(data, index=index)


def calibration_table(df: pd.DataFrame, col: str = "home_win_prob", bins: int = 10) -> pd.DataFrame:
    cut = pd.cut(df[col], bins=[i / bins for i in range(bins + 1)], include_lowest=True)
    return df.groupby(cut, observed=True).agg(n=("home_win", "size"), predicted=(col, "mean"), actual=("home_win", "mean"))
