from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from sportsbet.schedule import (
    ET,
    MAX_PULLS_PER_WEEK,
    due_slot,
    next_slot,
    pull_slots,
    should_pull_odds,
    week_bounds,
)


def _sched(games: list[tuple[str, str]]) -> pd.DataFrame:
    """Synthetic nflverse-style schedule from (gameday, gametime) pairs, times Eastern."""
    return pd.DataFrame(
        {
            "season": 2026,
            "week": 6,
            "game_type": "REG",
            "gameday": pd.to_datetime([d for d, _ in games]),
            "gametime": [t for _, t in games],
            "home_team": "BUF",
            "away_team": "NE",
            "result": None,
        }
    )


# Week of Tue 2026-10-06: Thu night, Sun early/late/night, Mon night. No Saturday.
REGULAR = _sched(
    [
        ("2026-10-08", "20:15"),
        ("2026-10-11", "13:00"),
        ("2026-10-11", "13:00"),
        ("2026-10-11", "16:25"),
        ("2026-10-11", "20:20"),
        ("2026-10-12", "20:15"),
    ]
)
# Same week with a Saturday game, which makes 7 candidate slots against a cap of 6.
WITH_SATURDAY = _sched([*zip(REGULAR["gameday"].dt.strftime("%Y-%m-%d"), REGULAR["gametime"]), ("2026-10-10", "16:30")])


def et(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=ET).astimezone(timezone.utc)


EXPECTED_REGULAR = {
    "Wed injury report": et("2026-10-07 12:00"),
    "Thu kickoff": et("2026-10-08 18:45"),
    "Fri injury report": et("2026-10-09 12:00"),
    "Sun early": et("2026-10-11 11:30"),
    "Sun late": et("2026-10-11 14:55"),
    "Mon kickoff": et("2026-10-12 18:45"),
}


def test_week_bounds_run_tuesday_to_tuesday_eastern():
    start, end = week_bounds(et("2026-10-11 13:00"))
    assert start == et("2026-10-06 00:00")
    assert end == et("2026-10-13 00:00")
    # Monday night after midnight UTC is still the same NFL week
    assert week_bounds(et("2026-10-12 23:30"))[0] == start


def test_slots_for_regular_week():
    slots = pull_slots(et("2026-10-06 09:00"), REGULAR)
    assert {s.label: s.at for s in slots} == EXPECTED_REGULAR
    assert [s.at for s in slots] == sorted(s.at for s in slots)


@pytest.mark.parametrize("label", list(EXPECTED_REGULAR))
def test_each_slot_pulls_inside_window_only(label):
    at = EXPECTED_REGULAR[label]
    assert should_pull_odds(at + timedelta(minutes=10), REGULAR)
    assert should_pull_odds(at - timedelta(minutes=29), REGULAR)
    assert due_slot(at + timedelta(minutes=59), REGULAR).label == label
    assert not should_pull_odds(at - timedelta(minutes=45), REGULAR)
    assert not should_pull_odds(at + timedelta(minutes=61), REGULAR)


def test_saturday_slot_replaces_wednesday_under_cap():
    slots = pull_slots(et("2026-10-06 09:00"), WITH_SATURDAY)
    labels = [s.label for s in slots]
    assert len(slots) == MAX_PULLS_PER_WEEK
    assert "Sat kickoff" in labels and "Wed injury report" not in labels
    assert should_pull_odds(et("2026-10-10 15:00"), WITH_SATURDAY)
    assert due_slot(et("2026-10-10 15:00"), WITH_SATURDAY).label == "Sat kickoff"
    assert not should_pull_odds(et("2026-10-07 12:10"), WITH_SATURDAY)


def test_pull_in_window_is_not_repeated():
    at = EXPECTED_REGULAR["Sun early"]
    first_tick = at - timedelta(minutes=20)
    assert should_pull_odds(first_tick, REGULAR, pull_log=[])
    assert not should_pull_odds(at + timedelta(minutes=40), REGULAR, pull_log=[first_tick])
    # a pull for the previous slot does not satisfy this one
    assert should_pull_odds(at, REGULAR, pull_log=[EXPECTED_REGULAR["Fri injury report"]])


def test_weekly_cap_blocks_even_an_open_window():
    at = EXPECTED_REGULAR["Mon kickoff"]
    forced = [et("2026-10-06 10:00") + timedelta(hours=i) for i in range(MAX_PULLS_PER_WEEK)]
    assert not should_pull_odds(at, REGULAR, pull_log=forced)
    assert should_pull_odds(at, REGULAR, pull_log=forced[:-1])
    # last week's pulls do not count against this week
    last_week = [p - timedelta(days=7) for p in forced]
    assert should_pull_odds(at, REGULAR, pull_log=last_week)


def test_no_games_means_no_pulls():
    assert not should_pull_odds(et("2027-03-10 12:00"), REGULAR)
    assert pull_slots(et("2027-03-10 12:00"), REGULAR) == []


def test_international_and_standard_time_kickoffs():
    # London 09:30 ET kickoff in November, after the switch to standard time (UTC-5)
    sched = _sched([("2026-11-08", "09:30"), ("2026-11-08", "13:00"), ("2026-11-08", "16:05")])
    slots = {s.label: s.at for s in pull_slots(et("2026-11-04 09:00"), sched)}
    assert slots["Sun early"] == datetime(2026, 11, 8, 13, 0, tzinfo=timezone.utc)
    assert slots["Sun late"] == et("2026-11-08 14:35")


def test_tbd_gametime_is_ignored():
    sched = _sched([("2026-10-11", "13:00"), ("2026-10-11", None)])
    assert [s.label for s in pull_slots(et("2026-10-06 09:00"), sched)][-1] == "Sun early"


def test_next_slot_looks_ahead():
    assert next_slot(et("2026-10-06 09:00"), REGULAR).label == "Wed injury report"
    assert next_slot(et("2026-10-11 12:00"), REGULAR).label == "Sun early"  # window still open
    assert next_slot(et("2026-10-12 21:00"), REGULAR) is None


def test_naive_datetime_rejected():
    with pytest.raises(ValueError):
        should_pull_odds(datetime(2026, 10, 11, 12), REGULAR)
