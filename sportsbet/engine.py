"""Find +EV prices at the target books by comparing against a de-vigged fair line.

Fair price hierarchy:
  1. Pinnacle's two-way market, power de-vigged.
  2. Consensus of the other reference books (mean implied prob per side), de-vigged.
For spreads and totals the same point is used when a reference quotes it. Otherwise
the reference line nearest the target point (Pinnacle first, then the consensus point
with the most books) is turned into a fair line with the push chart and priced at the
target point; fair_source then reads e.g. "pinnacle@-7.0". As with a same-point de-vig,
fair_prob on an integer point excludes pushes, so EV is per unit of action not refunded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from sportsbet import pushchart
from sportsbet.config import REFERENCE_BOOKS, SHARP_BOOK, TARGET_BOOKS
from sportsbet.pricing import devig_power, expected_value, implied_prob, kelly_fraction
from sportsbet.pushchart import PushChart

log = logging.getLogger(__name__)


@dataclass
class Candidate:
    event_id: str
    commence_time: Any
    matchup: str
    bookmaker: str
    market: str
    outcome: str
    point: float | None
    price: float
    fair_prob: float
    fair_source: str
    ev: float
    kelly: float
    model_prob: float | None = None

    def as_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["scanned_at"] = datetime.now(timezone.utc)
        return d


def _two_way(df: pd.DataFrame) -> tuple[pd.Series, pd.Series] | None:
    """Return the two rows of a two-outcome market, or None if not exactly two."""
    if len(df) != 2:
        return None
    a, b = df.iloc[0], df.iloc[1]
    return a, b


def fair_probs_for_market(
    market_rows: pd.DataFrame,
    sharp_book: str = SHARP_BOOK,
    reference_books: list[str] | None = None,
) -> tuple[dict[str, float], str] | None:
    """Map outcome -> fair probability for one (event, market, point) slice.

    Returns None when no reference book quotes both sides at this point.
    """
    reference_books = reference_books or REFERENCE_BOOKS
    sharp = market_rows[market_rows["bookmaker"] == sharp_book]
    pair = _two_way(sharp)
    if pair is not None:
        probs = devig_power([implied_prob(pair[0].price), implied_prob(pair[1].price)])
        return {pair[0].outcome: probs[0], pair[1].outcome: probs[1]}, sharp_book

    refs = market_rows[
        market_rows["bookmaker"].isin(reference_books) & ~market_rows["bookmaker"].isin(TARGET_BOOKS)
    ]
    if refs.empty:
        return None
    # keep only books quoting both sides so the consensus is not lopsided
    sides = refs.groupby("bookmaker")["outcome"].nunique()
    complete = refs[refs["bookmaker"].isin(sides[sides == 2].index)]
    if complete.empty:
        return None
    mean_probs = complete.assign(p=complete["price"].map(implied_prob)).groupby("outcome")["p"].mean()
    if len(mean_probs) != 2:
        return None
    fair = devig_power(list(mean_probs.values))
    n_books = complete["bookmaker"].nunique()
    return dict(zip(mean_probs.index, fair)), f"consensus({n_books})"


def _home_point(row: Any) -> float:
    """Spread point from the home team's side; totals are returned as is."""
    if row.market == "spreads" and row.outcome != row.home_team:
        return -float(row.point)
    return float(row.point)


def _reference_has_point(grp: pd.DataFrame, row: Any) -> bool:
    """Guard the abs(point) grouping: home -1 and home +1 share a key but are different bets."""
    if pd.isna(row.point):
        return True
    refs = grp[~grp["bookmaker"].isin(TARGET_BOOKS) & (grp["outcome"] == row.outcome)]
    return bool((refs["point"] == row.point).any())


def _reference_lines(ref: pd.DataFrame) -> list[tuple[tuple[int, int], float, float, str]]:
    """Two-way reference lines for one (event, market) as (priority, home_point, p, source).

    p is the de-vigged probability of the home side (spreads) or the over (totals).
    priority sorts Pinnacle first, then consensus lines with more books.
    """
    out = []
    for hp, grp in ref.groupby("home_point", sort=False):
        fair = fair_probs_for_market(grp)
        if fair is None:
            continue
        fair_map, source = fair
        first = grp.iloc[0]
        p = fair_map.get(first.home_team if first.market == "spreads" else "Over")
        if p is None:
            continue
        priority = (0, 0) if source == SHARP_BOOK else (1, -grp["bookmaker"].nunique())
        out.append((priority, float(hp), p, source))
    return out


def converted_fair_prob(
    ref: pd.DataFrame, row: Any, chart: PushChart
) -> tuple[float, str] | None:
    """Fair probability for a target spread/total row whose point no reference quotes.

    Chooses Pinnacle over consensus, then the reference point nearest the target, since
    the conversion error grows with the distance moved.
    """
    lines = _reference_lines(ref)
    if not lines:
        return None
    target_hp = _home_point(row)
    _, hp, p_first, source = min(lines, key=lambda t: (t[0], abs(t[1] - target_hp)))
    if row.market == "spreads":
        fair_line = chart.fair_spread_from_prob(hp, p_first)
        side = "home" if row.outcome == row.home_team else "away"
        p = chart.cover_prob(fair_line, float(row.point), side)
        ref_pt = hp if side == "home" else -hp
        return p, f"{source}@{ref_pt:+.1f}"
    fair_total = chart.fair_total_from_prob(hp, p_first)
    p_over = chart.over_prob(fair_total, float(row.point))
    return (p_over if row.outcome == "Over" else 1.0 - p_over), f"{source}@{hp:.1f}"


def _load_chart() -> PushChart | None:
    try:
        return pushchart.default_chart()
    except Exception as exc:  # the chart needs nflverse on first build; scan still works without it
        log.warning("push chart unavailable, cross-point spreads/totals skipped: %s", exc)
        return None


def scan(
    odds: pd.DataFrame,
    min_ev: float = 0.01,
    kelly_frac: float = 0.25,
    model_probs: dict[tuple[str, str], float] | None = None,
    push_chart: PushChart | None = None,
) -> list[Candidate]:
    """Scan a latest-odds frame for +EV prices at the target books.

    model_probs optionally maps (event_id, team_abbr) -> model win probability for h2h,
    reported alongside the market-based fair price as a second opinion.
    push_chart converts spreads/totals across points; the cached chart is loaded on
    first need when it is not given.
    """
    if odds.empty:
        return []
    odds = odds.copy()
    odds["point_key"] = odds["point"].fillna(0.0)
    # spreads are symmetric: home -3 pairs with away +3, so key spreads on abs(point)
    odds.loc[odds["market"] == "spreads", "point_key"] = odds.loc[odds["market"] == "spreads", "point"].abs()
    lined = odds["market"].isin(["spreads", "totals"]) & odds["point"].notna()
    odds["home_point"] = float("nan")
    odds.loc[lined, "home_point"] = [_home_point(r) for r in odds[lined].itertuples(index=False)]
    ref_rows = odds[lined & ~odds["bookmaker"].isin(TARGET_BOOKS)]
    refs_by_market = {k: g for k, g in ref_rows.groupby(["event_id", "market"], sort=False)}

    chart = push_chart
    chart_tried = push_chart is not None
    out: list[Candidate] = []
    for (event_id, market, _pk), grp in odds.groupby(["event_id", "market", "point_key"], sort=False):
        fair = fair_probs_for_market(grp)
        targets = grp[grp["bookmaker"].isin(TARGET_BOOKS)]
        if targets.empty:
            continue
        ref = refs_by_market.get((event_id, market))
        can_convert = market in ("spreads", "totals") and ref is not None
        if fair is None and not can_convert:
            continue
        first = grp.iloc[0]
        matchup = f"{first.away_team} @ {first.home_team}"
        for row in targets.itertuples(index=False):
            p = source = None
            if fair is not None and _reference_has_point(grp, row):
                p = fair[0].get(row.outcome)
                source = fair[1]
            if p is None and can_convert and pd.notna(row.point):
                if not chart_tried:
                    chart, chart_tried = _load_chart(), True
                conv = converted_fair_prob(ref, row, chart) if chart is not None else None
                if conv is not None:
                    p, source = conv
            if p is None:
                continue
            ev = expected_value(p, row.price)
            if ev < min_ev:
                continue
            mp = None
            if market == "h2h" and model_probs:
                mp = model_probs.get((event_id, row.outcome))
            out.append(
                Candidate(
                    event_id=event_id,
                    commence_time=first.commence_time,
                    matchup=matchup,
                    bookmaker=row.bookmaker,
                    market=market,
                    outcome=row.outcome,
                    point=None if pd.isna(row.point) else float(row.point),
                    price=float(row.price),
                    fair_prob=p,
                    fair_source=source,
                    ev=ev,
                    kelly=kelly_fraction(p, row.price, kelly_frac),
                    model_prob=mp,
                )
            )
    out.sort(key=lambda c: c.ev, reverse=True)
    return out


def format_candidates(cands: list[Candidate]) -> str:
    if not cands:
        return "No +EV prices found at the target books."
    rows = []
    for c in cands:
        pt = "" if c.point is None else f" {c.point:+g}" if c.market == "spreads" else f" {c.point:g}"
        model = "" if c.model_prob is None else f"{c.model_prob:.1%}"
        rows.append(
            {
                "game": c.matchup,
                "book": TARGET_BOOKS.get(c.bookmaker, c.bookmaker),
                "bet": f"{c.outcome}{pt}",
                "mkt": c.market,
                "price": f"{c.price:+.0f}",
                "fair": f"{c.fair_prob:.1%}",
                "src": c.fair_source,
                "EV": f"{c.ev:+.1%}",
                "kelly": f"{c.kelly:.2%}",
                "model": model,
            }
        )
    return pd.DataFrame(rows).to_string(index=False)
