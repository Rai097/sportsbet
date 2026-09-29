import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sportsbet.alerts import build_alerts, injury_changes, stale_lines
from sportsbet.model.injuries import STARTER_POINTS
from sportsbet.config import Settings
from sportsbet.poller import OddsBudget, Watcher, fixture_ticks
from sportsbet.providers.espn import InjuryRow, parse_espn_injuries
from sportsbet.providers.odds_api import OddsRow
from sportsbet.store import Store

NOW = datetime.now(timezone.utc).replace(microsecond=0)
KICKOFF = datetime(2099, 10, 4, 17, 0, tzinfo=timezone.utc)
STARTERS = {("BUF", "Josh Allen"), ("NE", "Starting Tackle")}


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.duckdb")
    yield s
    s.close()


def _injuries(at, allen_status, tackle_status="Out"):
    return [
        InjuryRow(at, "espn", "BUF", "Josh Allen", "QB", allen_status, "Shoulder"),
        InjuryRow(at, "espn", "BUF", "Some Backup", "WR", "Questionable", "Hamstring"),
        InjuryRow(at, "espn", "NE", "Starting Tackle", "OT", tackle_status, "Knee"),
    ]


def _spreads(at, book, home_point, price_home=-110, price_away=-110):
    common = dict(fetched_at=at, event_id="evt1", commence_time=KICKOFF, home_team="BUF", away_team="NE",
                  bookmaker=book, market="spreads", last_update=at)
    return [
        OddsRow(outcome="BUF", point=home_point, price=price_home, **common),
        OddsRow(outcome="NE", point=-home_point, price=price_away, **common),
    ]


def _seed(store, pinnacle_points=(-7.0, -7.0, -5.5)):
    """Allen Questionable at t-40m, Out at t-20m; odds at t-50m, t-30m, t-10m."""
    store.insert_rows("injuries", _injuries(NOW - timedelta(minutes=40), "Questionable"))
    store.insert_rows("injuries", _injuries(NOW - timedelta(minutes=20), "Out"))
    for mins, pin in zip((50, 30, 10), pinnacle_points):
        at = NOW - timedelta(minutes=mins)
        store.insert_rows("odds_snapshots", _spreads(at, "pinnacle", pin, -104, -106))
        store.insert_rows("odds_snapshots", _spreads(at, "betmgm", -7.0))


def test_injury_changes_flags_starter_qb(store):
    _seed(store)
    changes = injury_changes(store, starters=STARTERS)
    assert len(changes) == 1
    c = changes[0]
    assert (c.team, c.player, c.old_status, c.new_status, c.direction) == ("BUF", "Josh Allen", "Questionable", "Out", "worse")
    assert c.starter is True
    assert c.impact == pytest.approx(STARTER_POINTS["QB"] * (1.0 - 0.35))


def test_injury_changes_new_out_and_clearance(store):
    t0, t1 = NOW - timedelta(minutes=20), NOW - timedelta(minutes=10)
    store.insert_rows("injuries", _injuries(t0, "Questionable"))
    later = [r for r in _injuries(t1, "Questionable") if r.player != "Starting Tackle"]
    later.append(InjuryRow(t1, "espn", "NE", "New Guy", "CB", "Out", None))
    store.insert_rows("injuries", later)
    by_player = {c.player: c for c in injury_changes(store, starters=STARTERS)}
    assert by_player["New Guy"].direction == "worse" and by_player["New Guy"].old_status == "Not listed"
    assert by_player["New Guy"].starter is False  # not on the depth chart, so backup weighting
    assert by_player["Starting Tackle"].direction == "better" and by_player["Starting Tackle"].impact < 0
    assert "Josh Allen" not in by_player


def test_missing_team_block_is_not_a_clearance(store):
    store.insert_rows("injuries", _injuries(NOW - timedelta(minutes=20), "Out"))
    store.insert_rows("injuries", [r for r in _injuries(NOW - timedelta(minutes=10), "Out") if r.team == "BUF"])
    assert injury_changes(store) == []


def test_first_snapshot_is_baseline_not_news(store):
    store.insert_rows("injuries", _injuries(NOW, "Out"))
    assert injury_changes(store) == []


def test_stale_line_detects_pinnacle_move(store):
    _seed(store)
    stale = stale_lines(store, window_minutes=120)
    assert len(stale) == 1
    s = stale[0]
    assert (s.bookmaker, s.market, s.side, s.point, s.ref_source) == ("betmgm", "spreads", "NE", 7.0, "pinnacle")
    assert s.edge == pytest.approx(1.5)


def test_no_stale_line_when_target_follows(store):
    _seed(store)
    store.insert_rows("odds_snapshots", _spreads(NOW - timedelta(minutes=5), "betmgm", -5.5))
    store.insert_rows("odds_snapshots", _spreads(NOW - timedelta(minutes=5), "pinnacle", -5.5, -104, -106))
    assert stale_lines(store, window_minutes=120) == []


def test_high_priority_alert_and_dedup(store):
    _seed(store)
    alerts = build_alerts(store, starters=STARTERS, window_minutes=120)
    assert len(alerts) == 1  # the stale-line alert on the same bet is folded into the injury alert
    a = alerts[0]
    assert a.priority == "high"
    assert (a.bookmaker, a.market, a.side, a.point, a.price) == ("betmgm", "spreads", "NE", 7.0, -110)
    assert "Josh Allen" in a.detail and "Questionable -> Out" in a.detail
    assert build_alerts(store, starters=STARTERS, window_minutes=120) == []
    assert store.query("SELECT count(*) AS n FROM alerts")["n"].iloc[0] == 1


def test_no_injury_alert_once_target_repriced(store):
    _seed(store)
    store.insert_rows("odds_snapshots", _spreads(NOW - timedelta(minutes=5), "betmgm", -5.5))
    alerts = build_alerts(store, starters=STARTERS, window_minutes=120)
    assert not [a for a in alerts if a.priority == "high"]


def test_backup_qb_change_is_only_a_stale_line(store):
    _seed(store)
    alerts = build_alerts(store, starters={("BUF", "Someone Else")}, window_minutes=120)
    assert [a.priority for a in alerts] == ["medium"]
    assert alerts[0].trigger == "stale_line" and alerts[0].side == "NE"


def test_qb_change_priced_by_gap_like_the_report(store):
    """Alerts and the report must agree on what a QB is worth: his gap to the backup when known."""
    import pandas as pd

    from sportsbet.model.injuries import QB_POINTS_PER_VALUE, team_impacts

    _seed(store)
    gaps = {("BUF", "Josh Allen"): 6.0}
    c = injury_changes(store, starters=STARTERS, qb_gaps=gaps)[0]
    assert c.impact == pytest.approx(6.0 * QB_POINTS_PER_VALUE * (1.0 - 0.35), abs=1e-3)
    # the alert's impact is exactly the change in the report's team impact for that player
    times = store.injury_snapshot_times("espn")
    allen = [store.injury_snapshot(t, "espn").query("player == 'Josh Allen'") for t in times]
    before, after = (team_impacts(pd.DataFrame(a), STARTERS, qb_gaps=gaps)["BUF"].points for a in allen)
    assert c.impact == pytest.approx(after - before, abs=1e-3)

    # a backup nearly as good as the starter is not a high-priority injury
    alerts = build_alerts(store, starters=STARTERS, window_minutes=120, qb_gaps={("BUF", "Josh Allen"): 0.5})
    assert [a.priority for a in alerts] == ["medium"]


FIX_WATCH = Path(__file__).parent / "fixtures" / "watch"


def _watcher(store, **kw):
    lines: list[str] = []
    kw.setdefault("qb_gaps_loader", lambda: None)
    w = Watcher(store=store, settings=Settings(odds_api_key=None), starters_loader=lambda: {("BUF", "Josh Allen")},
                emit=lines.append, **kw)
    return w, lines


def test_poller_two_ticks_offline(store):
    w, lines = _watcher(store, fixtures=fixture_ticks(FIX_WATCH))
    assert w.run(interval=0, max_ticks=2) == 2
    alerts = store.query("SELECT * FROM alerts")
    high = alerts[alerts["priority"] == "high"]
    assert set(zip(high["bookmaker"], high["market"], high["side"])) == {("betmgm", "spreads", "NE"), ("betmgm", "h2h", "NE")}
    # Caesars repriced with Pinnacle, so it must not be flagged
    assert "williamhill_us" not in set(alerts["bookmaker"])
    assert any("[HIGH  ]" in line and "NE +7" in line for line in lines)
    # replaying the third (identical) tick yields nothing new
    w.fixtures = fixture_ticks(FIX_WATCH)[2:]
    w.run(interval=0)
    assert lines[-1].strip() == "no new alerts"
    assert len(store.query("SELECT * FROM alerts")) == len(alerts)


def test_poller_passes_qb_gaps_to_alerts(store):
    w, lines = _watcher(store, fixtures=fixture_ticks(FIX_WATCH), qb_gaps_loader=lambda: {("BUF", "Josh Allen"): 0.3})
    w.run(interval=0, max_ticks=2)
    alerts = store.query("SELECT * FROM alerts")
    assert "high" not in set(alerts["priority"])


def test_poller_without_api_key_skips_odds(store, monkeypatch):
    payload = json.loads((FIX_WATCH / "injuries_01.json").read_text())
    monkeypatch.setattr("sportsbet.poller.fetch_espn_injuries", lambda url: parse_espn_injuries(payload))
    w, lines = _watcher(store)
    assert w.run(interval=0, max_ticks=1) == 1
    assert "odds: skipped (no API key)" in lines[0]
    assert store.query("SELECT count(*) AS n FROM injuries")["n"].iloc[0] == 3


def test_poller_stops_cleanly_on_ctrl_c(store):
    w, lines = _watcher(store, fixtures=fixture_ticks(FIX_WATCH))

    def interrupt(_tick):
        raise KeyboardInterrupt

    w.tick = interrupt
    assert w.run(interval=0) == 0
    assert lines == ["stopped after 0 ticks"]


def test_odds_budget(store):
    budget = OddsBudget(every_seconds=3600, max_per_day=2)
    assert budget.blocked(store, NOW) is None
    store.insert_rows("odds_snapshots", _spreads(NOW - timedelta(minutes=30), "betmgm", -7.0))
    assert "last odds pull 30 min ago" in budget.blocked(store, NOW)
    store.insert_rows("odds_snapshots", _spreads(NOW - timedelta(hours=3), "betmgm", -7.0))
    assert "2 odds pulls in the last 24h" in budget.blocked(store, NOW + timedelta(hours=2))
    assert budget.blocked(store, NOW + timedelta(hours=22)) is None
