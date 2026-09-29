"""Find +EV prices at the target books by comparing against a de-vigged fair line.

Fair price hierarchy:
  1. Pinnacle's two-way market, power de-vigged.
  2. Consensus of the other reference books (mean implied prob per side), de-vigged.
For spreads and totals the comparison only happens at the same point; a half-point
difference is a different bet, and converting across points needs a push chart we
have not built yet.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from sportsbet.config import REFERENCE_BOOKS, SHARP_BOOK, TARGET_BOOKS
from sportsbet.pricing import devig_power, expected_value, implied_prob, kelly_fraction

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


def scan(
    odds: pd.DataFrame,
    min_ev: float = 0.01,
    kelly_frac: float = 0.25,
    model_probs: dict[tuple[str, str], float] | None = None,
) -> list[Candidate]:
    """Scan a latest-odds frame for +EV prices at the target books.

    model_probs optionally maps (event_id, team_abbr) -> model win probability for h2h,
    reported alongside the market-based fair price as a second opinion.
    """
    if odds.empty:
        return []
    odds = odds.copy()
    odds["point_key"] = odds["point"].fillna(0.0)
    # spreads are symmetric: home -3 pairs with away +3, so key spreads on abs(point)
    odds.loc[odds["market"] == "spreads", "point_key"] = odds.loc[odds["market"] == "spreads", "point"].abs()

    out: list[Candidate] = []
    for (event_id, market, _pk), grp in odds.groupby(["event_id", "market", "point_key"], sort=False):
        fair = fair_probs_for_market(grp)
        if fair is None:
            continue
        fair_map, source = fair
        first = grp.iloc[0]
        matchup = f"{first.away_team} @ {first.home_team}"
        targets = grp[grp["bookmaker"].isin(TARGET_BOOKS)]
        for row in targets.itertuples(index=False):
            p = fair_map.get(row.outcome)
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
