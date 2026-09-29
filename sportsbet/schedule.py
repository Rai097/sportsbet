"""Decide when an unattended run should spend Odds API credits.

The free tier is 500 credits a month and a pull costs 3, so pulls go where prices
matter most: about 90 minutes before the first kickoff of each slate day (plus the
Sunday late window), and once Wednesday and Friday midday Eastern to react to the
week's injury news. Everything else runs off free data.

The scheduler (GitHub Actions cron) fires hourly and can be late, so each slot is a
window rather than an instant, and a pull logged inside the window satisfies it. A
hard weekly cap bounds spend even if the schedule has more slots than expected or
someone forces extra pulls.

All functions here are pure: the caller passes the schedule and the pull log.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")
PRE_KICKOFF = timedelta(minutes=90)
# Hourly cron: a 90 minute window always contains at least one run, usually two.
WINDOW_BEFORE = timedelta(minutes=30)
WINDOW_AFTER = timedelta(minutes=60)
INJURY_SLOTS = {2: "Wed injury report", 4: "Fri injury report"}  # weekday -> label
INJURY_SLOT_TIME = time(16, 30)  # official reports usually publish mid to late afternoon ET
LATE_WINDOW_START = time(16, 0)
MAX_PULLS_PER_WEEK = 6
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@dataclass(frozen=True)
class PullSlot:
    at: datetime  # UTC
    label: str

    def window(self) -> tuple[datetime, datetime]:
        return self.at - WINDOW_BEFORE, self.at + WINDOW_AFTER


def _utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return ts.astimezone(timezone.utc)


def week_bounds(now_utc: datetime) -> tuple[datetime, datetime]:
    """The NFL week containing now: Tuesday 00:00 Eastern to the next Tuesday.

    Tuesday is the league's dead day, so every Thursday-to-Monday slate sits inside one window.
    """
    local = _utc(now_utc).astimezone(ET)
    start_day = local.date() - timedelta(days=(local.weekday() - 1) % 7)
    start = datetime.combine(start_day, time(0), tzinfo=ET)
    end = datetime.combine(start_day + timedelta(days=7), time(0), tzinfo=ET)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


def kickoffs_utc(schedule: pd.DataFrame) -> pd.Series:
    """Kickoff instants in UTC from nflverse `gameday` + `gametime` (Eastern). TBD times are dropped."""
    df = schedule.dropna(subset=["gameday", "gametime"])
    if df.empty:
        return pd.Series([], dtype="datetime64[ns, UTC]")
    local = pd.to_datetime(
        pd.to_datetime(df["gameday"]).dt.strftime("%Y-%m-%d") + " " + df["gametime"].astype(str),
        format="%Y-%m-%d %H:%M",
        errors="coerce",
    )
    local = local.dt.tz_localize(ET, ambiguous="NaT", nonexistent="shift_forward")
    return local.dt.tz_convert("UTC").dropna().sort_values()


def pull_slots(now_utc: datetime, schedule: pd.DataFrame, max_per_week: int = MAX_PULLS_PER_WEEK) -> list[PullSlot]:
    """Planned pulls for the NFL week containing now_utc, earliest first.

    No games that week (offseason, bye) means no slots. If the week has more slots than
    the cap allows (a Saturday slate), the injury-report slots are dropped first because
    a pre-kickoff price is the one that gets bet.
    """
    start, end = week_bounds(now_utc)
    ko = kickoffs_utc(schedule)
    ko = ko[(ko >= start) & (ko < end)]
    if ko.empty:
        return []
    kick_slots: list[PullSlot] = []
    by_day = ko.groupby(ko.dt.tz_convert(ET).dt.date)
    for day, times in by_day:
        first = times.min()
        dow = day.weekday()
        label = "Sun early" if dow == 6 else f"{DAY_NAMES[dow]} kickoff"
        kick_slots.append(PullSlot((first - PRE_KICKOFF).to_pydatetime(), label))
        if dow == 6:
            late = times[times.dt.tz_convert(ET).dt.time >= LATE_WINDOW_START]
            if not late.empty and late.min() != first:
                kick_slots.append(PullSlot((late.min() - PRE_KICKOFF).to_pydatetime(), "Sun late"))

    injury_slots: list[PullSlot] = []
    start_day = start.astimezone(ET).date()
    for offset in range(7):
        day = start_day + timedelta(days=offset)
        if day.weekday() not in INJURY_SLOTS:
            continue
        at = datetime.combine(day, INJURY_SLOT_TIME, tzinfo=ET).astimezone(timezone.utc)
        # a kickoff pull close by already captures the news, so do not pay twice
        if any(abs(s.at - at) < timedelta(hours=3) for s in kick_slots):
            continue
        injury_slots.append(PullSlot(at, INJURY_SLOTS[day.weekday()]))

    # drop Wednesday before Friday: Friday carries the final game statuses
    while injury_slots and len(kick_slots) + len(injury_slots) > max_per_week:
        injury_slots.pop(0)
    slots = sorted(kick_slots + injury_slots, key=lambda s: s.at)
    return slots[:max_per_week]


def pulls_this_week(now_utc: datetime, pull_log: Iterable[datetime]) -> int:
    start, _ = week_bounds(now_utc)
    now = _utc(now_utc)
    return sum(1 for t in pull_log if start <= _utc(t) <= now)


def due_slot(
    now_utc: datetime,
    schedule: pd.DataFrame,
    pull_log: Iterable[datetime] = (),
    max_per_week: int = MAX_PULLS_PER_WEEK,
) -> PullSlot | None:
    """The slot to pull for right now, or None if nothing is due or the weekly cap is spent."""
    now = _utc(now_utc)
    pulls = [_utc(t) for t in pull_log]
    if pulls_this_week(now, pulls) >= max_per_week:
        return None
    for slot in pull_slots(now, schedule, max_per_week):
        lo, hi = slot.window()
        if lo <= now < hi and not any(lo <= t <= now for t in pulls):
            return slot
    return None


def next_slot(now_utc: datetime, schedule: pd.DataFrame, max_per_week: int = MAX_PULLS_PER_WEEK) -> PullSlot | None:
    """First planned slot whose window has not closed yet, looking into next week if needed."""
    now = _utc(now_utc)
    for probe in (now, week_bounds(now)[1]):
        for slot in pull_slots(probe, schedule, max_per_week):
            if slot.window()[1] > now:
                return slot
    return None


def should_pull_odds(
    now_utc: datetime,
    season_schedule_df: pd.DataFrame,
    pull_log: Iterable[datetime] = (),
    max_per_week: int = MAX_PULLS_PER_WEEK,
) -> bool:
    """True when now falls in an unsatisfied pull window and the weekly cap has room."""
    return due_slot(now_utc, season_schedule_df, pull_log, max_per_week) is not None
