"""Polling loop behind `sportsbet watch`.

Each tick pulls ESPN's injuries feed (free) and stores a snapshot, pulls odds only when
both the odds interval and a rolling 24-hour pull budget allow (odds cost API credits),
then runs build_alerts and prints anything new. Alerts are deduplicated in the store,
so restarting the loop does not repeat them.

--fixture-dir replays recorded odds_*.json and injuries_*.json files in sorted order,
one pair per tick, so the loop can be exercised with no network and no credits.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

from sportsbet.alerts import Alert, add_alert_args, build_alerts, format_alerts, load_qb_gaps, load_starters
from sportsbet.config import Settings, load_settings
from sportsbet.providers.espn import fetch_espn_injuries, parse_espn_injuries
from sportsbet.providers.odds_api import OddsApiClient, load_fixture, normalize
from sportsbet.store import Store

log = logging.getLogger(__name__)

# 3 markets x one block of up to 10 books; see providers/odds_api.py
ODDS_PULL_COST = 3

FixtureTick = tuple[Path | None, Path | None]  # (odds file, injuries file)


def fixture_ticks(fixture_dir: Path) -> list[FixtureTick]:
    """Pair odds_*.json with injuries_*.json by sorted position; a shorter list leaves gaps."""
    odds = sorted(fixture_dir.glob("odds_*.json"))
    inj = sorted(fixture_dir.glob("injuries_*.json"))
    n = max(len(odds), len(inj))
    return [(odds[i] if i < len(odds) else None, inj[i] if i < len(inj) else None) for i in range(n)]


@dataclass
class OddsBudget:
    every_seconds: float = 3600.0
    max_per_day: int = 4

    def blocked(self, store: Store, now: datetime) -> str | None:
        """Why an odds pull is not allowed right now, or None.

        Counts pulls from the store rather than in memory so a restarted watcher, or a
        manual `sportsbet odds` run, is charged against the same budget.
        """
        pulls = store.query(
            "SELECT DISTINCT fetched_at FROM odds_snapshots WHERE fetched_at >= ?", [now - timedelta(days=1)]
        )["fetched_at"]
        if len(pulls) >= self.max_per_day:
            return f"{len(pulls)} odds pulls in the last 24h (max {self.max_per_day})"
        if len(pulls):
            since = (now - max(pulls).to_pydatetime()).total_seconds()
            if since < self.every_seconds:
                return f"last odds pull {since / 60:.0f} min ago (every {self.every_seconds / 60:.0f} min)"
        return None


@dataclass
class Watcher:
    store: Store
    settings: Settings
    budget: OddsBudget = field(default_factory=OddsBudget)
    fixtures: list[FixtureTick] | None = None
    client: OddsApiClient | None = None
    starters_loader: Callable[[], set[tuple[str, str]] | None] = load_starters
    qb_gaps_loader: Callable[[], dict[tuple[str, str], float] | None] = load_qb_gaps
    alert_kwargs: dict[str, Any] = field(default_factory=dict)
    emit: Callable[[str], None] = print
    _starters: set[tuple[str, str]] | None = None
    _starters_day: date | None = None
    _qb_gaps: dict[tuple[str, str], float] | None = None
    _warned_no_key: bool = False

    def starters(self, now: datetime) -> set[tuple[str, str]] | None:
        # depth charts change weekly at most; one load per day keeps ticks fast
        if self._starters_day != now.date():
            self._starters = self.starters_loader()
            self._qb_gaps = self.qb_gaps_loader()
            self._starters_day = now.date()
        return self._starters

    def pull_injuries(self, tick: int) -> str:
        if self.fixtures is not None:
            path = self.fixtures[tick][1]
            if path is None:
                return "injuries: no fixture"
            rows = parse_espn_injuries(json.loads(path.read_text()))
            label = path.name
        else:
            try:
                rows = fetch_espn_injuries(self.settings.espn_injuries_url)
            except requests.RequestException as exc:
                log.warning("ESPN injuries unavailable: %s", exc)
                return "injuries: fetch failed"
            label = "espn"
        n = self.store.insert_rows("injuries", rows)
        return f"injuries {n} rows ({label})"

    def pull_odds(self, tick: int, now: datetime) -> str:
        if self.fixtures is not None:
            # replayed pulls cost nothing, so the budget does not apply
            path = self.fixtures[tick][0]
            if path is None:
                return "odds: no fixture"
            rows = normalize(load_fixture(path))
            return f"odds {self.store.insert_rows('odds_snapshots', rows)} rows ({path.name})"
        if self.client is None:
            if not self._warned_no_key:
                log.warning("ODDS_API_KEY is not set: skipping odds pulls, alerting on stored odds only")
                self._warned_no_key = True
            return "odds: skipped (no API key)"
        reason = self.budget.blocked(self.store, now)
        if reason:
            return f"odds: skipped ({reason})"
        remaining = self.client.quota.remaining
        if remaining is not None and remaining - ODDS_PULL_COST < self.settings.quota_floor:
            return f"odds: skipped ({remaining} credits left, floor {self.settings.quota_floor})"
        try:
            rows = self.client.fetch_and_normalize()
        except (RuntimeError, requests.RequestException) as exc:  # QuotaExhausted is a RuntimeError
            log.warning("odds pull failed: %s", exc)
            return "odds: pull failed"
        n = self.store.insert_rows("odds_snapshots", rows)
        return f"odds {n} rows (credits remaining {self.client.quota.remaining})"

    def tick(self, tick: int) -> list[Alert]:
        now = datetime.now(timezone.utc)
        parts = [self.pull_injuries(tick), self.pull_odds(tick, now)]
        starters = self.starters(now)
        new = build_alerts(self.store, starters=starters, qb_gaps=self._qb_gaps, **self.alert_kwargs)
        self.emit(f"[{now:%Y-%m-%d %H:%M:%SZ}] tick {tick + 1}: " + "; ".join(parts))
        self.emit(format_alerts(new) if new else "  no new alerts")
        return new

    def run(self, interval: float, max_ticks: int | None = None) -> int:
        """Tick until max_ticks, the fixtures run out, or Ctrl-C. Returns ticks completed."""
        limit = max_ticks
        if self.fixtures is not None:
            limit = len(self.fixtures) if limit is None else min(limit, len(self.fixtures))
        done = 0
        try:
            while limit is None or done < limit:
                self.tick(done)
                done += 1
                if limit is None or done < limit:
                    time.sleep(interval)
        except KeyboardInterrupt:
            self.emit(f"stopped after {done} ticks")
        return done


def cmd_watch(args) -> int:
    """Poll injuries (and odds within budget) and print new alerts after every tick."""
    settings = load_settings()
    fixtures = None
    if args.fixture_dir:
        fixtures = fixture_ticks(Path(args.fixture_dir))
        if not fixtures:
            print(f"No odds_*.json or injuries_*.json files in {args.fixture_dir}")
            return 1
    # replayed fixtures go to a throwaway DB unless asked, so they never mix with real snapshots
    store = Store(args.db or (":memory:" if fixtures is not None else settings.db_path))
    client = OddsApiClient(settings) if fixtures is None and settings.odds_api_key else None
    watcher = Watcher(
        store=store,
        settings=settings,
        budget=OddsBudget(args.odds_every, args.max_odds_pulls_per_day),
        fixtures=fixtures,
        client=client,
        starters_loader=(lambda: None) if args.no_depth_charts else load_starters,
        qb_gaps_loader=(lambda: None) if args.no_depth_charts else load_qb_gaps,
        alert_kwargs={"window_minutes": args.window, "injury_lookback_hours": args.lookback, "min_impact": args.min_impact},
    )
    try:
        watcher.run(args.interval, args.max_ticks)
    finally:
        store.close()
    return 0


def register(subparsers) -> None:
    s = subparsers.add_parser("watch", help=cmd_watch.__doc__)
    s.add_argument("--interval", type=float, default=300, help="seconds between ticks")
    s.add_argument("--odds-every", type=float, default=3600, help="minimum seconds between odds pulls")
    s.add_argument("--max-odds-pulls-per-day", type=int, default=4, help="odds pulls allowed in any 24h")
    s.add_argument("--fixture-dir", help="replay odds_*.json / injuries_*.json pairs, one per tick, offline")
    s.add_argument("--max-ticks", type=int, help="stop after this many ticks")
    add_alert_args(s)
    s.set_defaults(func=cmd_watch)
