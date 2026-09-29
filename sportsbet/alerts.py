"""Injury status-change alerts and stale-line detection.

The edge this chases is latency. When ESPN reports a starter's status change, or
Pinnacle moves, BetMGM and Caesars often take a while to reprice. Everything here works
off stored snapshots (the `injuries` and `odds_snapshots` tables), so it spends no API
credits and can run after every poll.

Priorities:
  high    a starter-level injury change (|impact| >= 1 point) on team T where a target
          book's line on T's game has not moved since the injury snapshot.
  medium  the reference (Pinnacle, else the consensus) moved and a target book did not.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from sportsbet.config import REFERENCE_BOOKS, SHARP_BOOK, TARGET_BOOKS, load_settings
from sportsbet.model.injuries import BACKUP_FRACTION, STARTER_POINTS, STATUS_WEIGHT, starters_from_depth_chart
from sportsbet.pricing import devig_power, expected_value, implied_prob
from sportsbet.store import Store

log = logging.getLogger(__name__)

# A reference move at least this big is treated as news (points, or win probability for h2h).
MOVE_THRESHOLD = {"spreads": 0.5, "totals": 0.5, "h2h": 0.025}
# A target book that moved less than this has not repriced. Spreads and totals move in
# half points, so anything under 0.5 means the number itself is unchanged.
STILL_THRESHOLD = {"spreads": 0.5, "totals": 0.5, "h2h": 0.01}
WORSE_STATUSES = {"Doubtful", "Out"}
OFF_REPORT = "Not listed"
MIN_IMPACT = 1.0
# Totals barely react to most injuries, so injury alerts only look at sides.
INJURY_MARKETS = ("spreads", "h2h")
DEFAULT_WINDOW_MINUTES = 720.0
DEFAULT_LOOKBACK_HOURS = 24.0


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _fmt_ts(ts: Any) -> str:
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%a %H:%MZ")


@dataclass
class InjuryChange:
    fetched_at: datetime  # snapshot where the new status first appeared
    prev_fetched_at: datetime
    source: str
    team: str
    player: str
    position: str | None
    old_status: str
    new_status: str
    direction: str  # worse | better
    starter: bool | None  # None when depth charts were unavailable
    impact: float  # spread points; positive means the team got weaker

    def describe(self) -> str:
        role = "" if self.starter is None else ", starter" if self.starter else ", backup"
        return (
            f"{self.player} ({self.team} {self.position or '?'}{role}) "
            f"{self.old_status} -> {self.new_status}, {self.impact:+.1f} pts"
        )


@dataclass
class StaleLine:
    event_id: str
    commence_time: Any
    matchup: str
    bookmaker: str
    market: str
    side: str  # team abbreviation, or Over / Under
    point: float | None
    price: float
    ref_source: str
    ref_from: float
    ref_to: float
    target_from: float
    target_to: float
    edge: float  # points (spreads/totals) or win probability (h2h) in the side's favour
    first_at: Any
    last_at: Any


@dataclass
class Alert:
    priority: str  # high | medium
    trigger: str
    event_id: str
    commence_time: Any
    matchup: str
    bookmaker: str
    market: str
    side: str
    point: float | None
    price: float
    detail: str
    alert_key: str
    created_at: datetime = field(default_factory=_utcnow)

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _weight(status: str) -> float:
    return STATUS_WEIGHT.get(status, 0.0)


def _classify(old: str, new: str) -> str | None:
    w_old, w_new = _weight(old), _weight(new)
    if w_new > w_old and new in WORSE_STATUSES:
        return "worse"
    if w_new < w_old:
        return "better"
    return None


def _diff_snapshots(
    prev: pd.DataFrame,
    cur: pd.DataFrame,
    starters: set[tuple[str, str]] | None,
    source: str,
    prev_at: datetime,
    cur_at: datetime,
) -> list[InjuryChange]:
    prev_rows = {(r.team, r.player): r for r in prev.itertuples(index=False)}
    cur_rows = {(r.team, r.player): r for r in cur.itertuples(index=False)}
    teams_now = {t for t, _ in cur_rows}
    out: list[InjuryChange] = []
    for key in sorted(prev_rows.keys() | cur_rows.keys()):
        team, player = key
        if key not in cur_rows and team not in teams_now:
            # the whole team block is missing from the feed, which is not a clearance
            continue
        old = prev_rows[key].status if key in prev_rows else OFF_REPORT
        new = cur_rows[key].status if key in cur_rows else OFF_REPORT
        direction = _classify(old, new)
        if direction is None:
            continue
        row = cur_rows.get(key) or prev_rows[key]
        pos = row.position if isinstance(row.position, str) else None
        starter = None if starters is None else key in starters
        base = STARTER_POINTS.get((pos or "").upper(), 0.0)
        impact = base * (_weight(new) - _weight(old)) * (BACKUP_FRACTION if starter is False else 1.0)
        out.append(
            InjuryChange(
                fetched_at=cur_at,
                prev_fetched_at=prev_at,
                source=source,
                team=team,
                player=player,
                position=pos,
                old_status=old,
                new_status=new,
                direction=direction,
                starter=starter,
                impact=round(impact, 3),
            )
        )
    return out


def injury_changes(
    store: Store,
    since: datetime | None = None,
    starters: set[tuple[str, str]] | None = None,
    source: str | None = None,
) -> list[InjuryChange]:
    """Status changes between consecutive injury snapshots, largest impact first.

    since=None compares the latest snapshot with the previous one. With a timestamp,
    every consecutive pair whose newer snapshot is after `since` is diffed, so a change
    stays visible for a while even when injuries are polled far more often than odds.
    Snapshots are compared per source because ESPN and the official report differ in
    naming and coverage.
    """
    sources = [source] if source else store.query("SELECT DISTINCT source FROM injuries")["source"].tolist()
    out: list[InjuryChange] = []
    for src in sources:
        times = store.injury_snapshot_times(src)
        if len(times) < 2:
            continue  # the first snapshot is a baseline, not news
        if since is None:
            out += _diff_snapshots(
                store.previous_injuries(src), store.injury_snapshot(times[-1], src), starters, src, times[-2], times[-1]
            )
            continue
        for prev_at, cur_at in zip(times, times[1:]):
            if cur_at > since:
                out += _diff_snapshots(
                    store.injury_snapshot(prev_at, src), store.injury_snapshot(cur_at, src), starters, src, prev_at, cur_at
                )
    out.sort(key=lambda c: (-abs(c.impact), c.team, c.player))
    return out


def _book_value(rows: pd.DataFrame, market: str, home: str) -> float | None:
    """One number per book per snapshot: home spread point, total, or de-vigged home win prob."""
    if market == "spreads":
        r = rows[rows["outcome"] == home]["point"].dropna()
        return float(r.iloc[0]) if len(r) else None
    if market == "totals":
        r = rows[rows["outcome"] == "Over"]["point"].dropna()
        return float(r.iloc[0]) if len(r) else None
    if len(rows) != 2 or home not in set(rows["outcome"]):
        return None
    probs = devig_power([implied_prob(p) for p in rows["price"]])
    return float(probs[list(rows["outcome"]).index(home)])


def _consensus_value(rows: pd.DataFrame, market: str, home: str) -> float | None:
    books = [b for b in REFERENCE_BOOKS if b != SHARP_BOOK and b not in TARGET_BOOKS]
    vals = [_book_value(rows[rows["bookmaker"] == b], market, home) for b in books]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _series(snaps: dict[Any, pd.DataFrame], market: str, home: str, book: str | None) -> dict[Any, float]:
    """Value per snapshot time for one book, or the consensus when book is None."""
    out = {}
    for t, g in snaps.items():
        v = _consensus_value(g, market, home) if book is None else _book_value(g[g["bookmaker"] == book], market, home)
        if v is not None:
            out[t] = v
    return out


def _favourable(market: str, ref_move: float, ref_now: float, tgt_now: float, home: str, away: str) -> tuple[str, float]:
    """Which side the target book's stale number favours, and by how much."""
    if market == "spreads":  # values are the home point: more negative = home more favoured
        return (home, tgt_now - ref_now) if ref_move < 0 else (away, ref_now - tgt_now)
    if market == "totals":
        return ("Over", ref_now - tgt_now) if ref_move > 0 else ("Under", tgt_now - ref_now)
    return (home, ref_now - tgt_now) if ref_move > 0 else (away, tgt_now - ref_now)


def _side_row(rows: pd.DataFrame, side: str) -> pd.Series | None:
    r = rows[rows["outcome"] == side]
    return r.iloc[0] if len(r) else None


def _point(row: pd.Series) -> float | None:
    return None if pd.isna(row["point"]) else float(row["point"])


def stale_lines(
    store: Store,
    window_minutes: float = DEFAULT_WINDOW_MINUTES,
    now: datetime | None = None,
    history: pd.DataFrame | None = None,
) -> list[StaleLine]:
    """Target-book lines that sat still while the reference moved within the window.

    The comparison runs from the first to the last snapshot in the window where both
    the reference and the target quote the market. Pinnacle is the reference whenever it
    quoted at both ends; otherwise the mean of the other reference books is used.
    """
    now = now or _utcnow()
    hist = history if history is not None else store.odds_history(hours=window_minutes / 60.0, now=now)
    if hist.empty:
        return []
    hist = hist[hist["fetched_at"] >= pd.Timestamp(now - timedelta(minutes=window_minutes))]
    out: list[StaleLine] = []
    for (event_id, market), grp in hist.groupby(["event_id", "market"], sort=False):
        if market not in MOVE_THRESHOLD:
            continue
        first = grp.iloc[0]
        home, away = first.home_team, first.away_team
        snaps = {t: g for t, g in grp.groupby("fetched_at")}
        refs = [(SHARP_BOOK, _series(snaps, market, home, SHARP_BOOK)), ("consensus", _series(snaps, market, home, None))]
        for book in TARGET_BOOKS:
            tgt = _series(snaps, market, home, book)
            for ref_source, ref in refs:
                common = sorted(t for t in tgt if t in ref)
                if len(common) >= 2:
                    break
            else:
                continue
            t0, t1 = common[0], common[-1]
            ref_move = ref[t1] - ref[t0]
            if abs(ref_move) < MOVE_THRESHOLD[market] or abs(tgt[t1] - tgt[t0]) >= STILL_THRESHOLD[market]:
                continue
            side, edge = _favourable(market, ref_move, ref[t1], tgt[t1], home, away)
            last = snaps[t1]
            row = _side_row(last[last["bookmaker"] == book], side)
            if edge <= 1e-9 or row is None:
                continue
            out.append(
                StaleLine(
                    event_id=event_id,
                    commence_time=first.commence_time,
                    matchup=f"{away} @ {home}",
                    bookmaker=book,
                    market=market,
                    side=side,
                    point=_point(row),
                    price=float(row["price"]),
                    ref_source=ref_source,
                    ref_from=ref[t0],
                    ref_to=ref[t1],
                    target_from=tgt[t0],
                    target_to=tgt[t1],
                    edge=edge,
                    first_at=t0,
                    last_at=t1,
                )
            )
    return out


def _value_label(market: str, home: str, v: float) -> str:
    if market == "spreads":
        return f"{home} {v:+g}"
    if market == "totals":
        return f"{v:g}"
    return f"{home} {v:.1%}"


def _stale_alert(s: StaleLine) -> Alert:
    home = s.matchup.split(" @ ")[1]
    ref_txt = f"{s.ref_source} {_value_label(s.market, home, s.ref_from)} -> {_value_label(s.market, home, s.ref_to)}"
    tgt_txt = f"{s.bookmaker} still {_value_label(s.market, home, s.target_to)}"
    if s.market == "h2h":
        fair = s.ref_to if s.side == home else 1.0 - s.ref_to
        edge_txt = f"fair {fair:.1%}, EV {expected_value(fair, s.price):+.1%}"
    else:
        edge_txt = f"{s.edge:.1f} pts better than the reference"
    ref_key = f"{s.ref_source}:{s.ref_to:.4f}"
    return Alert(
        priority="medium",
        trigger="stale_line",
        event_id=s.event_id,
        commence_time=s.commence_time,
        matchup=s.matchup,
        bookmaker=s.bookmaker,
        market=s.market,
        side=s.side,
        point=s.point,
        price=s.price,
        detail=f"{ref_txt} since {_fmt_ts(s.first_at)}; {tgt_txt}; {edge_txt}",
        alert_key="|".join([s.event_id, s.bookmaker, s.market, s.side, "stale_line", ref_key]),
    )


def _injury_alerts(
    changes: list[InjuryChange],
    hist: pd.DataFrame,
    min_impact: float,
) -> list[Alert]:
    # a player can change twice inside the lookback; only the latest status matters
    latest: dict[tuple[str, str], InjuryChange] = {}
    for c in sorted(changes, key=lambda c: c.fetched_at):
        latest[(c.team, c.player)] = c
    out: list[Alert] = []
    for c in latest.values():
        if abs(c.impact) < min_impact:
            continue
        games = hist[(hist["home_team"] == c.team) | (hist["away_team"] == c.team)]
        if games.empty:
            continue
        event_id = games.sort_values("commence_time")["event_id"].iloc[0]
        game = games[games["event_id"] == event_id]
        home, away = game["home_team"].iloc[0], game["away_team"].iloc[0]
        opponent = away if c.team == home else home
        side = opponent if c.direction == "worse" else c.team
        injury_at = pd.Timestamp(c.fetched_at)
        for market in INJURY_MARKETS:
            mrows = game[game["market"] == market]
            for book in TARGET_BOOKS:
                snaps = {t: g for t, g in mrows[mrows["bookmaker"] == book].groupby("fetched_at")}
                before = [t for t in snaps if t <= injury_at]
                if not before:
                    continue  # no line from before the news, so we cannot tell if it moved
                t0, t1 = max(before), max(snaps)
                v0, v1 = _book_value(snaps[t0], market, home), _book_value(snaps[t1], market, home)
                row = _side_row(snaps[t1], side)
                if v0 is None or v1 is None or row is None or abs(v1 - v0) >= STILL_THRESHOLD[market]:
                    continue
                note = f"{book} unchanged since {_fmt_ts(t0)}"
                if t1 == t0:
                    note += " (no odds pull since the injury snapshot)"
                sharp = _series({t: g for t, g in mrows.groupby("fetched_at")}, market, home, SHARP_BOOK)
                sharp_after = [t for t in sharp if t >= t0]
                if len(sharp_after) >= 2 and sharp[max(sharp_after)] != sharp[min(sharp_after)]:
                    a, b = sharp[min(sharp_after)], sharp[max(sharp_after)]
                    note += f"; {SHARP_BOOK} {_value_label(market, home, a)} -> {_value_label(market, home, b)}"
                out.append(
                    Alert(
                        priority="high",
                        trigger=f"injury:{c.team}:{c.player}",
                        event_id=event_id,
                        commence_time=game["commence_time"].iloc[0],
                        matchup=f"{away} @ {home}",
                        bookmaker=book,
                        market=market,
                        side=side,
                        point=_point(row),
                        price=float(row["price"]),
                        detail=f"{c.describe()}; {note}",
                        alert_key="|".join(
                            [event_id, book, market, side, f"injury:{c.team}:{c.player}", injury_at.isoformat()]
                        ),
                    )
                )
    return out


def build_alerts(
    store: Store,
    starters: set[tuple[str, str]] | None = None,
    window_minutes: float = DEFAULT_WINDOW_MINUTES,
    injury_lookback_hours: float = DEFAULT_LOOKBACK_HOURS,
    min_impact: float = MIN_IMPACT,
    now: datetime | None = None,
    persist: bool = True,
) -> list[Alert]:
    """Alerts not emitted before, high priority first. Persists them unless persist=False."""
    now = now or _utcnow()
    changes = injury_changes(store, since=now - timedelta(hours=injury_lookback_hours), starters=starters)
    hist = store.odds_history(hours=injury_lookback_hours + window_minutes / 60.0, now=now)
    alerts = _injury_alerts(changes, hist, min_impact) if not hist.empty else []
    covered = {(a.event_id, a.bookmaker, a.market, a.side) for a in alerts}
    for s in stale_lines(store, window_minutes, now=now, history=hist):
        # an injury alert on the same bet already says the line is stale, with the reason
        if (s.event_id, s.bookmaker, s.market, s.side) not in covered:
            alerts.append(_stale_alert(s))
    seen = store.recent_alert_keys()
    new: list[Alert] = []
    for a in alerts:
        if a.alert_key not in seen:
            seen.add(a.alert_key)
            new.append(a)
    new.sort(key=lambda a: (a.priority != "high", str(a.commence_time), a.matchup, a.bookmaker, a.market))
    if persist and new:
        store.insert_alerts(new)
    return new


def _side_label(a: Alert) -> str:
    if a.market == "spreads":
        return f"{a.side} {a.point:+g}" if a.point is not None else a.side
    if a.market == "totals":
        return f"{a.side} {a.point:g}" if a.point is not None else a.side
    return f"{a.side} ML"


def format_alerts(alerts: list[Alert]) -> str:
    if not alerts:
        return "No new alerts."
    lines = []
    for a in alerts:
        book = TARGET_BOOKS.get(a.bookmaker, a.bookmaker)
        lines.append(
            f"[{a.priority.upper():<6}] {a.matchup:<10} {book:<7} {a.market:<7} "
            f"{_side_label(a)} ({a.price:+.0f})  {a.detail}"
        )
    return "\n".join(lines)


def load_starters() -> set[tuple[str, str]] | None:
    """Current depth-chart starters from nflverse, or None (everyone counts as a starter)."""
    from sportsbet.providers import nflverse

    try:
        season = nflverse.current_season()
        return starters_from_depth_chart(nflverse.load_depth_charts([season]), season)
    except Exception as exc:  # depth charts are best-effort, same as the scan
        log.warning("depth charts unavailable, treating every listed player as a starter: %s", exc)
        return None


def cmd_alerts(args) -> int:
    """Build alerts from stored injury and odds snapshots and print the new ones."""
    settings = load_settings()
    store = Store(args.db or settings.db_path)
    try:
        starters = None if args.no_depth_charts else load_starters()
        new = build_alerts(
            store,
            starters=starters,
            window_minutes=args.window,
            injury_lookback_hours=args.lookback,
            min_impact=args.min_impact,
        )
        print(format_alerts(new))
    finally:
        store.close()
    return 0


def add_alert_args(s: argparse.ArgumentParser) -> None:
    s.add_argument("--window", type=float, default=DEFAULT_WINDOW_MINUTES, help="minutes of odds history for stale lines")
    s.add_argument("--lookback", type=float, default=DEFAULT_LOOKBACK_HOURS, help="hours of injury snapshots to diff")
    s.add_argument("--min-impact", type=float, default=MIN_IMPACT, help="points of impact for a high-priority alert")
    s.add_argument("--no-depth-charts", action="store_true", help="skip nflverse depth charts; every player counts as a starter")
    s.add_argument("--db", help="DuckDB path (default from settings)")


def register(subparsers) -> None:
    s = subparsers.add_parser("alerts", help=cmd_alerts.__doc__)
    add_alert_args(s)
    s.set_defaults(func=cmd_alerts)
