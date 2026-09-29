import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from sportsbet import report
from sportsbet.config import Settings
from sportsbet.providers.odds_api import normalize
from sportsbet.store import Store

FIX = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 10, 7, 14, 0, tzinfo=timezone.utc)  # Wednesday of week 6


def history() -> pd.DataFrame:
    """Two seasons of synthetic nflverse schedule: 2025 played, 2026 weeks 5 played and 6 pending."""
    rows = []
    for i, day in enumerate(pd.date_range("2025-09-07", periods=6, freq="7D")):
        rows.append((2025, i + 1, day, "13:00", "NE", "BUF", 17, 27))
        rows.append((2025, i + 1, day, "16:25", "KC", "LV", 30, 20))
    rows.append((2026, 5, pd.Timestamp("2026-10-04"), "13:00", "BUF", "NE", 24, 21))
    rows.append((2026, 6, pd.Timestamp("2026-10-08"), "20:15", "LV", "KC", None, None))
    rows.append((2026, 6, pd.Timestamp("2026-10-11"), "13:00", "NE", "BUF", None, None))
    df = pd.DataFrame(
        rows, columns=["season", "week", "gameday", "gametime", "away_team", "home_team", "away_score", "home_score"]
    )
    df["game_type"] = "REG"
    df["location"] = "Home"
    df["result"] = df["home_score"] - df["away_score"]
    df["game_id"] = [f"g{i}" for i in range(len(df))]
    df["spread_line"] = 3.0
    df["total_line"] = 45.5
    df["home_moneyline"] = -150
    df["away_moneyline"] = 130
    df["home_qb_name"] = "QB1"
    df["away_qb_name"] = "QB2"
    return df


def official_injuries(_seasons) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2026, 2026],
            "week": [6, 6],
            "team": ["BUF", "NE"],
            "full_name": ["Josh Allen", "Starting Tackle"],
            "position": ["QB", "T"],
            "report_status": ["Out", "Questionable"],
        }
    )


def depth_charts(_seasons) -> pd.DataFrame:
    return pd.DataFrame(
        {"dt": ["2026-10-06"] * 2, "team": ["BUF", "NE"], "player_name": ["Josh Allen", "Starting Tackle"], "pos_rank": [1, 1]}
    )


def offline(*_a, **_k):
    raise ConnectionError("network blocked in test")


def fixture_rows():
    return normalize(json.loads((FIX / "odds_api_nfl_sample.json").read_text()))


@pytest.fixture
def settings(tmp_path):
    return Settings(odds_api_key=None, data_dir=tmp_path, db_path=tmp_path / "t.duckdb")


@pytest.fixture
def free_data(monkeypatch):
    monkeypatch.setattr(report.nflverse, "load_schedules", lambda seasons: history()[lambda d: d["season"].isin(seasons)])
    monkeypatch.setattr(report.nflverse, "load_injuries", official_injuries)
    monkeypatch.setattr(report.nflverse, "load_depth_charts", depth_charts)
    # the QB model needs nflverse player stats; tests that want gaps patch this themselves
    monkeypatch.setattr(report, "load_qb_gaps", offline)


@pytest.fixture
def no_network(monkeypatch):
    for name in ("load_schedules", "load_injuries", "load_depth_charts"):
        monkeypatch.setattr(report.nflverse, name, offline)


def test_full_report(settings, free_data, tmp_path):
    store = Store(settings.db_path)
    store.insert_rows("odds_snapshots", fixture_rows())
    data = report.gather(store, settings, now=NOW, min_ev=0.0)
    md = report.render(data)
    assert data.season == 2026 and data.week == 6
    assert "# NFL +EV report: 2026 week 6" in md
    assert "NE @ BUF" in md and "LV @ KC" in md
    assert "BetMGM" in md and "pinnacle" in md
    assert "Josh Allen (QB, Out) -3.8" in md
    assert "official report, week 6" in md
    assert data.candidates and any(c.model_prob is not None for c in data.candidates)
    assert "ODDS_API_KEY: **not set**" in md
    assert "Next planned pull: Wed injury report" in md

    path = report.write_report(md, data, tmp_path / "reports")
    assert path == tmp_path / "reports" / "2026-wk06" / "20261007T140000Z.md"
    assert (tmp_path / "reports" / "latest.md").read_text() == md


def test_empty_store_and_no_network(settings, no_network, tmp_path):
    store = Store(settings.db_path)
    data = report.gather(store, settings, now=NOW)
    md = report.render(data)
    assert data.week is None
    assert "Not available: could not load the nflverse schedule" in md
    assert "Not available: no odds for upcoming games are stored" in md
    assert "## Injury impact" in md and "Not available" in md
    assert "Latest odds snapshot: never" in md
    path = report.write_report(md, data, tmp_path / "reports")
    assert path.parent.name == "2026-wkNA" and path.exists()


def test_model_failure_falls_back_to_market_only(settings, free_data, monkeypatch):
    def broken_fit(self, games):
        raise RuntimeError("elo exploded")

    monkeypatch.setattr(report.EloModel, "fit", broken_fit)
    store = Store(settings.db_path)
    store.insert_rows("odds_snapshots", fixture_rows())
    data = report.gather(store, settings, now=NOW, min_ev=0.0)
    md = report.render(data)
    assert data.candidates and all(c.model_prob is None for c in data.candidates)
    assert "Model unavailable (elo exploded)" in md
    assert "NE @ BUF" in md  # slate still shows the market line
    assert "| kickoff (ET) | game | market |" in md


def test_espn_snapshot_preferred_when_fresh(settings, free_data):
    from sportsbet.providers.espn import parse_espn_injuries

    store = Store(settings.db_path)
    rows = parse_espn_injuries(json.loads((FIX / "espn_injuries_sample.json").read_text()), fetched_at=NOW)
    store.insert_rows("injuries", rows)
    data = report.gather(store, settings, now=NOW)
    assert data.injuries_source.startswith("ESPN live feed")


def test_quota_section_reads_pull_log(settings, free_data):
    store = Store(settings.db_path)
    store.log_pull(NOW, "Wed injury report", False, 24, 3, 12, 488)
    data = report.gather(store, settings, now=NOW)
    md = report.render(data)
    assert "API credits remaining 488" in md
    assert "Pulls this NFL week: 1 of 6" in md
    assert "Pulls logged this month: 1 (~3 credits)" in md


def test_unchanged_report_is_not_rewritten(settings, free_data, tmp_path):
    store = Store(settings.db_path)
    out = tmp_path / "reports"
    first = report.gather(store, settings, now=NOW)
    assert report.write_report(report.render(first), first, out, only_if_changed=True)
    later = report.gather(store, settings, now=NOW.replace(minute=30))
    assert report.write_report(report.render(later), later, out, only_if_changed=True) is None
    assert report.write_report(report.render(later), later, out) is not None


def test_scan_and_report_agree_on_model_probs(settings, free_data, monkeypatch):
    """cli.scan and the report price injuries the same way, QB value gap included."""
    from sportsbet import cli

    monkeypatch.setattr(report, "load_qb_gaps", lambda season, sched, depth: {("BUF", "Josh Allen"): 2.0})
    monkeypatch.setattr(report, "utcnow", lambda: NOW)
    store = Store(settings.db_path)
    store.insert_rows("odds_snapshots", fixture_rows())
    data = report.gather(store, settings, now=NOW, min_ev=-1.0)
    # 2.0 value gap x 0.9 points per value, not the flat 3.8
    assert "Josh Allen (QB, Out) -1.8" in report.render(data)
    from_report = {(c.event_id, c.outcome): c.model_prob for c in data.candidates if c.model_prob is not None}
    from_cli = cli._model_probs(store, store.latest_odds(), use_injuries=True)
    assert from_report and all(from_cli[k] == pytest.approx(v) for k, v in from_report.items())
    no_inj = cli._model_probs(store, store.latest_odds(), use_injuries=False)
    assert no_inj[("evt1", "BUF")] > from_cli[("evt1", "BUF")]  # Allen out costs BUF


def test_report_shows_recent_alerts_only(settings, free_data):
    from datetime import timedelta

    from sportsbet.alerts import Alert

    kickoff = datetime(2099, 10, 4, 17, tzinfo=timezone.utc)

    def alert(key, created, commence=kickoff, market="h2h", point=None):
        return Alert(priority="high", trigger="injury:BUF:Josh Allen", event_id="evt1", commence_time=commence,
                     matchup="NE @ BUF", bookmaker="betmgm", market=market, side="NE", point=point, price=285.0,
                     detail=f"detail {key}", alert_key=key, created_at=created)

    store = Store(settings.db_path)
    store.insert_alerts([
        alert("fresh", NOW - timedelta(hours=1)),
        alert("fresh-spread", NOW - timedelta(hours=2), market="spreads", point=7.0),
        alert("old", NOW - timedelta(days=2)),
        alert("started", NOW - timedelta(hours=1), commence=NOW - timedelta(minutes=5)),
    ])
    md = report.render(report.gather(store, settings, now=NOW))
    section = md.split("## Alerts (last 24h)")[1].split("## This week's slate")[0]
    assert "NE ML (+285)  detail fresh" in section and "NE +7 (+285)  detail fresh-spread" in section
    assert "detail old" not in section and "detail started" not in section

    empty = report.render(report.gather(Store(settings.data_dir / "empty.duckdb"), settings, now=NOW))
    assert "No alerts in the last 24 hours." in empty
