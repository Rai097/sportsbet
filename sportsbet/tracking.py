"""Bet tracking: log placed bets, capture the closing line, settle from nflverse, report CLV.

Closing line value is the only short-term signal that separates edge from luck, so every
bet records the fair probability when it was placed and the fair probability at the close.

Definitions used in the report:
  CLV (pp)     closing_fair_prob - fair_prob at bet, in probability points, for the side bet.
               Positive means the sharp market moved toward the bet after it was placed.
  CLV (cents)  price taken vs the no-vig closing price, both on a continuous American scale
               where +100 and -100 coincide (-110 -> -10, +120 -> +20). Positive means the
               price taken was better than the fair close.
  beat close   the price taken implies a lower probability than the no-vig close, i.e. the
               bet was still +EV at kickoff.
  EV@close     expected value per unit of the price taken if the closing fair prob is true.

Extra per-bet fields (kickoff time, closing point, notes) live in the bet_meta side table
rather than new columns on bets, so the bets schema stays exactly as other code expects it.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from sportsbet.config import TARGET_BOOKS, load_settings
from sportsbet.engine import fair_probs_for_market
from sportsbet.pricing import american_to_decimal, expected_value, implied_prob
from sportsbet.providers import nflverse
from sportsbet.store import Store
from sportsbet.teams import to_abbr

MARKET_ALIASES = {"ml": "h2h", "moneyline": "h2h", "spread": "spreads", "total": "totals"}
BOOK_ALIASES = {"mgm": "betmgm", "caesars": "williamhill_us", "czr": "williamhill_us"}
MIN_MEANINGFUL_N = 100


@dataclass
class Bet:
    bet_id: str
    placed_at: datetime
    event_id: str | None
    matchup: str
    bookmaker: str
    market: str
    outcome: str
    point: float | None
    price: float
    stake: float
    fair_prob: float | None
    closing_price: float | None = None
    closing_fair_prob: float | None = None
    result: str | None = None
    pnl: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


# --- normalisation -----------------------------------------------------------------------


def normalize_book(book: str) -> str:
    key = book.strip().lower()
    by_name = {name.lower(): k for k, name in TARGET_BOOKS.items()}
    return BOOK_ALIASES.get(key) or by_name.get(key) or key


def normalize_market(market: str) -> str:
    key = market.strip().lower()
    return MARKET_ALIASES.get(key, key)


def normalize_outcome(outcome: str) -> str:
    if outcome.strip().lower() in ("over", "under"):
        return outcome.strip().capitalize()
    try:
        return to_abbr(outcome.strip())
    except KeyError:
        return to_abbr(outcome.strip().upper())


def parse_matchup(matchup: str) -> tuple[str, str]:
    """'NE @ BUF' -> ('NE', 'BUF') as (away, home)."""
    if "@" not in matchup:
        raise ValueError(f"matchup must look like 'AWAY @ HOME', got {matchup!r}")
    away, home = (s.strip() for s in matchup.split("@", 1))
    return normalize_outcome(away), normalize_outcome(home)


def _none_if_nan(v: Any) -> Any:
    if v is None:
        return None
    try:
        return None if pd.isna(v) else v
    except (TypeError, ValueError):
        return v


def _utc(ts: Any) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


# --- pricing helpers ---------------------------------------------------------------------


def market_slice(rows: pd.DataFrame, market: str, outcome: str, point: float | None) -> pd.DataFrame:
    """Rows of one two-way market at the bet's point (both sides), across all books."""
    rows = rows[rows["market"] == market]
    if market == "spreads" and point is not None:
        same = (rows["outcome"] == outcome) & (rows["point"] == point)
        other = (rows["outcome"] != outcome) & (rows["point"] == -point)
        return rows[same | other]
    if market == "totals" and point is not None:
        return rows[rows["point"] == point]
    return rows


def fair_prob_for_bet(
    snapshot: pd.DataFrame, market: str, outcome: str, point: float | None
) -> tuple[float, str] | None:
    """Reference fair prob of the bet's side in one odds pull, or None if no reference quotes it."""
    if snapshot.empty:
        return None
    res = fair_probs_for_market(market_slice(snapshot, market, outcome, point))
    if res is None or outcome not in res[0]:
        return None
    return float(res[0][outcome]), res[1]


def american_exact(p: float) -> float:
    """Unrounded American odds for a probability (pricing.prob_to_american rounds)."""
    dec = 1.0 / p
    return (dec - 1.0) * 100.0 if dec >= 2.0 else -100.0 / (dec - 1.0)


def cents(american: float) -> float:
    """Map American odds to a continuous scale so price differences are in 'cents'."""
    return american - 100.0 if american >= 100 else american + 100.0


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


# --- add ---------------------------------------------------------------------------------


def _resolve_event(store: Store, away: str, home: str, at: pd.Timestamp) -> tuple[str, pd.Timestamp] | None:
    """Stored event for this matchup: the next kickoff after `at`, else the most recent one."""
    ev = store.query(
        """
        SELECT DISTINCT event_id, commence_time FROM odds_snapshots
        WHERE away_team = ? AND home_team = ?
        """,
        [away, home],
    )
    if ev.empty:
        return None
    ev["commence_time"] = ev["commence_time"].map(_utc)
    upcoming = ev[ev["commence_time"] >= at].sort_values("commence_time")
    row = upcoming.iloc[0] if not upcoming.empty else ev.sort_values("commence_time").iloc[-1]
    return row["event_id"], row["commence_time"]


def _latest_candidate(store: Store, event_id: str, book: str, market: str, outcome: str, point: float | None):
    df = store.query(
        """
        SELECT * FROM candidates
        WHERE event_id = ? AND bookmaker = ? AND market = ? AND outcome = ?
          AND point IS NOT DISTINCT FROM ?
        ORDER BY scanned_at DESC LIMIT 1
        """,
        [event_id, book, market, outcome, point],
    )
    return None if df.empty else df.iloc[0]


def add_bet(
    store: Store,
    *,
    bookmaker: str,
    market: str,
    outcome: str,
    point: float | None = None,
    price: float | None = None,
    stake: float = 1.0,
    matchup: str | None = None,
    event_id: str | None = None,
    commence_time: Any = None,
    placed_at: Any = None,
) -> Bet:
    """Record a bet. Price and fair prob come from the latest matching scan candidate when one
    exists; otherwise price is required and fair prob is de-vigged from the last stored pull.

    commence_time is only needed for bets on games with no stored odds (settlement uses it
    to find the game in the nflverse schedule).
    """
    placed = _utc(placed_at or datetime.now(timezone.utc))
    book, mkt, side = normalize_book(bookmaker), normalize_market(market), normalize_outcome(outcome)
    if mkt not in ("h2h", "spreads", "totals"):
        raise ValueError(f"unsupported market {market!r}")
    if mkt != "h2h" and point is None:
        raise ValueError(f"{mkt} bets need a point")
    point = None if mkt == "h2h" else float(point)

    if event_id:
        found = store.query(
            "SELECT away_team, home_team, max(commence_time) AS commence_time FROM odds_snapshots "
            "WHERE event_id = ? GROUP BY 1, 2",
            [event_id],
        )
        if found.empty:
            raise ValueError(f"no stored odds for event {event_id}")
        away, home, commence = found.iloc[0]["away_team"], found.iloc[0]["home_team"], _utc(found.iloc[0]["commence_time"])
    else:
        if not matchup:
            raise ValueError("need matchup ('AWAY @ HOME') or event_id")
        away, home = parse_matchup(matchup)
        resolved = _resolve_event(store, away, home, placed)
        if resolved is not None:
            event_id, commence = resolved
        elif commence_time is not None:
            commence = _utc(commence_time)
        else:
            raise ValueError(f"no stored odds for {away} @ {home}; pass a kickoff date to log it anyway")
    if mkt != "totals" and side not in (away, home):
        raise ValueError(f"{side} is not in {away} @ {home}")

    fair_prob, fair_source, source = None, None, "manual"
    cand = _latest_candidate(store, event_id, book, mkt, side, point) if event_id else None
    if cand is not None:
        source = "scan"
        fair_prob, fair_source = float(cand["fair_prob"]), cand["fair_source"]
        price = float(cand["price"]) if price is None else float(price)
    else:
        if price is None:
            raise ValueError("no scan candidate matches this bet; pass the price you got")
        if event_id:
            fair = fair_prob_for_bet(store.last_snapshot_before(event_id, placed), mkt, side, point)
            if fair is not None:
                fair_prob, fair_source = fair

    bet = Bet(
        bet_id=uuid.uuid4().hex[:8],
        placed_at=placed.to_pydatetime(),
        event_id=event_id,
        matchup=f"{away} @ {home}",
        bookmaker=book,
        market=mkt,
        outcome=side,
        point=point,
        price=float(price),
        stake=float(stake),
        fair_prob=fair_prob,
    )
    store.insert_rows("bets", [bet])
    store.update_bet(bet.bet_id, commence_time=commence.to_pydatetime(), source=source, fair_source=fair_source)
    return bet


# --- closing line ------------------------------------------------------------------------


def capture_closing(store: Store, now: Any = None) -> int:
    """Fill closing price/point/fair prob for open bets whose game has kicked off.

    Closing price and point are the book's last quote on that side before kickoff. The
    closing fair prob is de-vigged from the last pull before kickoff at the bet's own
    point; if the reference no longer quotes that point it stays null with a note, because
    a fair prob at a different number is a different bet.
    """
    now = _utc(now or datetime.now(timezone.utc))
    captured = 0
    for b in store.open_bets().itertuples(index=False):
        commence = _none_if_nan(b.commence_time)
        if commence is None or _utc(commence) > now or _none_if_nan(b.closing_fetched_at) is not None:
            continue
        if _none_if_nan(b.event_id) is None:
            store.update_bet(b.bet_id, note="no stored odds for this game; no closing line")
            continue
        commence = _utc(commence).to_pydatetime()
        point = _none_if_nan(b.point)
        notes: list[str] = []
        fields: dict[str, Any] = {}

        quote = store.last_quote_before(b.event_id, b.bookmaker, b.market, b.outcome, commence)
        if quote.empty:
            notes.append("book had no quote before kickoff")
        else:
            q = quote.iloc[0]
            fields["closing_price"] = float(q["price"])
            fields["closing_point"] = _none_if_nan(q["point"])
            if b.market != "h2h" and fields["closing_point"] != point:
                notes.append(f"book closed at {fields['closing_point']:g}, bet at {point:g}")

        snap = store.last_snapshot_before(b.event_id, commence)
        fair = fair_prob_for_bet(snap, b.market, b.outcome, point)
        if fair is not None:
            fields["closing_fair_prob"], fields["closing_fair_source"] = fair
        elif snap.empty:
            notes.append("no odds pull before kickoff")
        else:
            notes.append(f"no reference line at {point:g} in last pre-kickoff pull" if point is not None
                         else "no reference line in last pre-kickoff pull")
        if not snap.empty:
            fields["closing_fetched_at"] = _utc(snap["fetched_at"].iloc[0]).to_pydatetime()
        fields["note"] = "; ".join(notes) or None
        store.update_bet(b.bet_id, **fields)
        captured += 1
    return captured


# --- settlement --------------------------------------------------------------------------


def grade(market: str, outcome: str, point: float | None, home: str, away: str,
          home_margin: float, total: float) -> str:
    """'win', 'loss' or 'push'. A moneyline tie is a push, as BetMGM and Caesars grade it."""
    if market == "h2h":
        diff = home_margin if outcome == home else -home_margin
    elif market == "spreads":
        diff = (home_margin if outcome == home else -home_margin) + point
    elif market == "totals":
        diff = total - point if outcome == "Over" else point - total
    else:
        raise ValueError(f"cannot grade market {market!r}")
    return "win" if diff > 0 else "loss" if diff < 0 else "push"


def bet_pnl(result: str, price: float, stake: float) -> float:
    if result == "win":
        return stake * (american_to_decimal(price) - 1.0)
    if result == "loss":
        return -stake
    return 0.0


def _find_game(sched: pd.DataFrame, home: str, away: str, commence: pd.Timestamp):
    """Schedule row for the game plus the sign that maps its result to our home team's margin.

    Matches on teams and US-Eastern kickoff date +-1 day; retries with teams swapped because
    neutral-site games can list a different designated home team than the odds feed.
    """
    day = commence.tz_convert("America/New_York").tz_localize(None).normalize()
    gameday = pd.to_datetime(sched["gameday"])
    near = (gameday - day).abs() <= pd.Timedelta(days=1)
    for h, a, sign in ((home, away, 1.0), (away, home, -1.0)):
        g = sched[near & (sched["home_team"] == h) & (sched["away_team"] == a)]
        if not g.empty:
            return g.iloc[0], sign
    return None, 0.0


def settle(store: Store, now: Any = None, schedule: pd.DataFrame | None = None) -> list[str]:
    """Grade open bets on finished games. Returns the settled bet ids."""
    now = _utc(now or datetime.now(timezone.utc))
    open_ = store.open_bets()
    if open_.empty:
        return []
    open_ = open_[open_["commence_time"].notna()]
    open_ = open_[open_["commence_time"].map(_utc) <= now]
    if open_.empty:
        return []
    if schedule is None:
        seasons = sorted({nflverse.current_season(_utc(t).date()) for t in open_["commence_time"]})
        schedule = nflverse.load_schedules(seasons)
    settled = []
    for b in open_.itertuples(index=False):
        away, home = parse_matchup(b.matchup)
        game, sign = _find_game(schedule, home, away, _utc(b.commence_time))
        if game is None or pd.isna(game["result"]):
            continue
        total = game["total"] if "total" in game.index and pd.notna(game["total"]) else game["home_score"] + game["away_score"]
        result = grade(b.market, b.outcome, _none_if_nan(b.point), home, away, sign * float(game["result"]), float(total))
        store.update_bet(b.bet_id, result=result, pnl=bet_pnl(result, b.price, b.stake))
        settled.append(b.bet_id)
    return settled


# --- reporting ---------------------------------------------------------------------------


def bet_metrics(bets: pd.DataFrame) -> pd.DataFrame:
    """Per-bet CLV columns (NaN where the closing fair prob is unknown)."""
    df = bets.copy()
    has_close = df["closing_fair_prob"].notna()
    df["clv_prob"] = df["closing_fair_prob"] - df["fair_prob"]
    df["clv_cents"] = float("nan")
    df["beat_close"] = float("nan")
    df["ev_close"] = float("nan")
    for i in df.index[has_close]:
        cf, price = float(df.at[i, "closing_fair_prob"]), float(df.at[i, "price"])
        df.at[i, "clv_cents"] = cents(price) - cents(american_exact(cf))
        df.at[i, "beat_close"] = float(implied_prob(price) < cf)
        df.at[i, "ev_close"] = expected_value(cf, price)
    return df


def _summarize(df: pd.DataFrame) -> dict[str, Any]:
    done = df[df["result"].notna()]
    stake_done = float(done["stake"].sum())
    return {
        "bets": len(df),
        "settled": len(done),
        "stake": float(df["stake"].sum()),
        "pnl": float(done["pnl"].sum()),
        "roi": float(done["pnl"].sum()) / stake_done if stake_done else float("nan"),
        "clv_n": int(df["closing_fair_prob"].notna().sum()),
        "clv_pp": float(df["clv_prob"].mean() * 100) if df["clv_prob"].notna().any() else float("nan"),
        "clv_cents": float(df["clv_cents"].mean()) if df["clv_cents"].notna().any() else float("nan"),
        "beat_close": float(df["beat_close"].mean()) if df["beat_close"].notna().any() else float("nan"),
        "ev_close": float(df["ev_close"].mean()) if df["ev_close"].notna().any() else float("nan"),
    }


def clv_report(store: Store) -> pd.DataFrame:
    """One row per book, per market, and overall. ROI is pnl over settled stake."""
    bets = store.bets_with_meta()
    if bets.empty:
        return pd.DataFrame()
    df = bet_metrics(bets)
    rows = []
    for key in ("bookmaker", "market"):
        for val, grp in df.groupby(key, sort=True):
            label = TARGET_BOOKS.get(val, val) if key == "bookmaker" else val
            rows.append({"by": "book" if key == "bookmaker" else "market", "group": label, **_summarize(grp)})
    rows.append({"by": "all", "group": "overall", **_summarize(df)})
    return pd.DataFrame(rows)


def sample_size_note(report: pd.DataFrame) -> str:
    overall = report[report["by"] == "all"].iloc[0]
    n, settled = int(overall["clv_n"]), int(overall["settled"])
    parts = []
    if n == 0:
        parts.append("No bets have a closing reference yet.")
    else:
        k = round(overall["beat_close"] * n)
        lo, hi = wilson_interval(k, n)
        parts.append(f"{k}/{n} bets beat the no-vig close ({k / n:.0%}, 95% Wilson interval {lo:.0%}-{hi:.0%}).")
    if settled:
        # a unit bet near even money has a standard deviation of about 1 unit
        parts.append(f"ROI standard error at near-even odds is about +-{1 / math.sqrt(settled):.0%} at {settled} settled bets.")
    if max(n, settled) < MIN_MEANINGFUL_N:
        parts.append(f"n < {MIN_MEANINGFUL_N}: this means nothing yet; even at n=100 a coin flip spans roughly 40-60%.")
    return " ".join(parts)


def format_report(report: pd.DataFrame) -> str:
    if report.empty:
        return "No bets logged yet. Use `sportsbet bet add`."

    def pct(v: float, signed: bool = False) -> str:
        return "" if pd.isna(v) else f"{v:+.1%}" if signed else f"{v:.0%}"

    def num(v: float, fmt: str) -> str:
        return "" if pd.isna(v) else format(v, fmt)

    out = pd.DataFrame(
        {
            "by": report["by"],
            "group": report["group"],
            "bets": report["bets"],
            "settled": report["settled"],
            "stake": report["stake"].map(lambda v: f"{v:g}"),
            "pnl": report["pnl"].map(lambda v: f"{v:+.2f}"),
            "ROI": report["roi"].map(lambda v: pct(v, signed=True)),
            "clv_n": report["clv_n"],
            "CLV_pp": report["clv_pp"].map(lambda v: num(v, "+.2f")),
            "CLV_cents": report["clv_cents"].map(lambda v: num(v, "+.1f")),
            "beat_close": report["beat_close"].map(pct),
            "EV@close": report["ev_close"].map(lambda v: pct(v, signed=True)),
        }
    )
    return out.to_string(index=False) + "\n\n" + sample_size_note(report)


def _fmt_point(market: str, point: Any) -> str:
    point = _none_if_nan(point)
    if point is None:
        return ""
    return f" {point:+g}" if market == "spreads" else f" {point:g}"


def format_bets(bets: pd.DataFrame) -> str:
    if bets.empty:
        return "No bets."

    def prob(v: Any) -> str:
        return "" if pd.isna(v) else f"{v:.1%}"

    rows = []
    for b in bets.itertuples(index=False):
        close = ""
        if not pd.isna(b.closing_price):
            cp = _fmt_point(b.market, b.closing_point) if not pd.isna(b.closing_point) else ""
            close = f"{cp.strip()} {b.closing_price:+.0f}".strip()
        rows.append(
            {
                "id": b.bet_id,
                "placed_utc": _utc(b.placed_at).strftime("%Y-%m-%d %H:%M"),
                "game": b.matchup,
                "book": TARGET_BOOKS.get(b.bookmaker, b.bookmaker),
                "bet": f"{b.outcome}{_fmt_point(b.market, b.point)}",
                "mkt": b.market,
                "price": f"{b.price:+.0f}",
                "stake": f"{b.stake:g}",
                "fair": prob(b.fair_prob),
                "close": close,
                "close_fair": prob(b.closing_fair_prob),
                "result": "" if pd.isna(b.result) else b.result,
                "pnl": "" if pd.isna(b.pnl) else f"{b.pnl:+.2f}",
                "note": "" if pd.isna(b.note) else b.note,
            }
        )
    return pd.DataFrame(rows).to_string(index=False)


# --- CLI ---------------------------------------------------------------------------------


def _open_store() -> Store:
    return Store(load_settings().db_path)


def cmd_bet_add(args) -> int:
    """Log a bet you placed, from the latest scan candidate or from a manual price."""
    store = _open_store()
    try:
        bet = add_bet(
            store,
            bookmaker=args.book,
            market=args.market,
            outcome=args.outcome,
            point=args.point,
            price=args.price,
            stake=args.stake,
            matchup=args.game,
            event_id=args.event_id,
            commence_time=pd.Timestamp(args.date, tz="America/New_York") + pd.Timedelta(hours=13) if args.date else None,
        )
    except ValueError as exc:
        print(f"error: {exc}")
        store.close()
        return 1
    fair = "no fair prob" if bet.fair_prob is None else f"fair {bet.fair_prob:.1%}"
    print(f"added {bet.bet_id}: {bet.matchup} {TARGET_BOOKS.get(bet.bookmaker, bet.bookmaker)} "
          f"{bet.outcome}{_fmt_point(bet.market, bet.point)} {bet.market} {bet.price:+.0f} stake {bet.stake:g} ({fair})")
    store.close()
    return 0


def cmd_bet_list(args) -> int:
    """List logged bets."""
    store = _open_store()
    print(format_bets(store.bets_with_meta(open_only=args.open)))
    store.close()
    return 0


def cmd_bet_settle(args) -> int:
    """Capture closing lines for started games, then grade finished games from nflverse."""
    store = _open_store()
    n_close = capture_closing(store)
    settled = settle(store)
    print(f"captured closing lines for {n_close} bets; settled {len(settled)} bets")
    if settled:
        bets = store.bets_with_meta()
        print(format_bets(bets[bets["bet_id"].isin(settled)]))
    store.close()
    return 0


def cmd_clv(args) -> int:
    """Closing line value and P&L by book, market and overall."""
    store = _open_store()
    print(format_report(clv_report(store)))
    store.close()
    return 0


def register(subparsers: Any) -> None:
    s = subparsers.add_parser("bet", help="log, list and settle placed bets")
    bet_sub = s.add_subparsers(dest="bet_cmd", required=True)

    a = bet_sub.add_parser("add", help=cmd_bet_add.__doc__)
    a.add_argument("--game", help="'AWAY @ HOME' abbreviations, e.g. 'NE @ BUF'")
    a.add_argument("--event-id", help="Odds API event id instead of --game")
    a.add_argument("--book", required=True, help="betmgm | caesars | any Odds API bookmaker key")
    a.add_argument("--market", required=True, help="h2h | spreads | totals (ml, spread, total accepted)")
    a.add_argument("--outcome", required=True, help="team abbreviation, or Over / Under")
    a.add_argument("--point", type=float, help="spread for the side bet (e.g. -7) or total line")
    a.add_argument("--price", type=float, help="American odds taken (default: latest scan price)")
    a.add_argument("--stake", type=float, default=1.0, help="stake in units")
    a.add_argument("--date", help="kickoff date YYYY-MM-DD, only for games with no stored odds")
    a.set_defaults(func=cmd_bet_add)

    ls = bet_sub.add_parser("list", help=cmd_bet_list.__doc__)
    ls.add_argument("--open", action="store_true", help="only unsettled bets")
    ls.set_defaults(func=cmd_bet_list)

    st = bet_sub.add_parser("settle", help=cmd_bet_settle.__doc__)
    st.set_defaults(func=cmd_bet_settle)

    c = subparsers.add_parser("clv", help=cmd_clv.__doc__)
    c.set_defaults(func=cmd_clv)

