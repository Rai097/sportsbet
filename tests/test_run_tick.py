import json
from datetime import datetime, timedelta, timezone

import pytest
from test_report import FIX, fixture_rows, history, offline

from sportsbet import cli, report
from sportsbet.config import Settings
from sportsbet.providers.espn import parse_espn_injuries
from sportsbet.providers.odds_api import OddsApiClient
from sportsbet.store import Store

# 90 minutes before Thursday's 20:15 ET kickoff in the synthetic schedule, plus a few minutes of cron lag
THU_SLOT = datetime(2026, 10, 8, 22, 50, tzinfo=timezone.utc)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Offline world: synthetic nflverse data, clock inside a pull window, reports in tmp."""
    settings = Settings(odds_api_key=None, data_dir=tmp_path, db_path=tmp_path / "t.duckdb")
    monkeypatch.setattr(report, "load_settings", lambda: settings)
    monkeypatch.setattr(report, "utcnow", lambda: THU_SLOT)
    monkeypatch.setattr(report.nflverse, "load_schedules", lambda seasons: history()[lambda d: d["season"].isin(seasons)])
    monkeypatch.setattr(report.nflverse, "load_injuries", offline)
    monkeypatch.setattr(report.nflverse, "load_depth_charts", offline)
    monkeypatch.setattr(report, "fetch_espn_injuries", offline)
    calls = []

    def fake_fetch(self):
        calls.append(self.settings.odds_api_key)
        self.quota.remaining, self.quota.used, self.quota.last_cost = 497, 3, 3
        return fixture_rows()

    monkeypatch.setattr(OddsApiClient, "fetch_and_normalize", fake_fetch)
    return settings, tmp_path / "reports", calls


def run(out, *extra) -> int:
    return cli.main(["run-tick", "--out", str(out), "--min-ev", "0", *extra])


def test_no_key_skips_pull_and_still_reports(env, capsys):
    settings, out, calls = env
    assert run(out) == 0
    printed = capsys.readouterr().out
    assert "odds pull skipped (Thu kickoff): ODDS_API_KEY is not set" in printed
    assert "ESPN injuries skipped" in printed
    assert calls == []
    latest = (out / "latest.md").read_text()
    assert "Not available: no odds for upcoming games are stored" in latest
    assert list((out / "2026-wk06").glob("*.md"))


def test_scheduled_pull_logs_scans_and_reports(env, monkeypatch, capsys):
    settings, out, calls = env
    settings.odds_api_key = "test-key"
    espn = parse_espn_injuries(json.loads((FIX / "espn_injuries_sample.json").read_text()), fetched_at=THU_SLOT)
    monkeypatch.setattr(report, "fetch_espn_injuries", lambda url: espn)

    assert run(out) == 0
    assert calls == ["test-key"]
    store = Store(settings.db_path)
    log = store.pull_log()
    assert list(log["slot"]) == ["Thu kickoff"] and int(log["credits_remaining"].iloc[0]) == 497
    assert len(store.query("SELECT * FROM candidates")) > 0
    assert len(store.latest_injury_snapshot("espn")) == 3
    store.close()
    latest = (out / "latest.md").read_text()
    assert "BetMGM" in latest and "API credits remaining 497" in latest
    assert "ESPN live feed" in latest
    assert "alerts: 0 new" in capsys.readouterr().out  # one odds pull is a baseline, not news
    files = list((out / "2026-wk06").glob("*.md"))
    assert len(files) == 1

    # next hourly tick inside the same window: no second pull, nothing new to write
    monkeypatch.setattr(report, "utcnow", lambda: THU_SLOT + timedelta(minutes=50))
    capsys.readouterr()
    assert run(out) == 0
    printed = capsys.readouterr().out
    assert calls == ["test-key"]
    assert "no pull window open" in printed
    assert "report: unchanged" in printed


def test_force_pull_outside_window(env, monkeypatch):
    settings, out, calls = env
    settings.odds_api_key = "test-key"
    monkeypatch.setattr(report, "utcnow", lambda: THU_SLOT - timedelta(days=1, hours=5))
    assert run(out) == 0
    assert calls == []
    assert run(out, "--force-pull") == 0
    assert calls == ["test-key"]
    store = Store(settings.db_path)
    assert store.pull_log()["forced"].tolist() == [True]
    store.close()


def test_schedule_outage_still_exits_zero(env, monkeypatch, capsys):
    settings, out, calls = env
    settings.odds_api_key = "test-key"
    monkeypatch.setattr(report.nflverse, "load_schedules", offline)
    assert run(out) == 0
    assert calls == []
    assert "schedule unavailable" in capsys.readouterr().out
    assert (out / "latest.md").exists()


def test_tick_builds_alerts_into_the_report(env, capsys):
    """Unattended runs never call `alerts`, so run-tick must build them for the report to show."""
    from test_alerts import _injuries, _spreads

    settings, out, calls = env
    store = Store(settings.db_path)
    store.insert_rows("injuries", _injuries(THU_SLOT - timedelta(minutes=40), "Questionable"))
    store.insert_rows("injuries", _injuries(THU_SLOT - timedelta(minutes=20), "Out"))
    for mins, pin in ((50, -7.0), (10, -5.5)):
        at = THU_SLOT - timedelta(minutes=mins)
        store.insert_rows("odds_snapshots", _spreads(at, "pinnacle", pin, -104, -106))
        store.insert_rows("odds_snapshots", _spreads(at, "betmgm", -7.0))
    store.close()

    assert run(out) == 0
    assert "alerts: 1 new" in capsys.readouterr().out
    latest = (out / "latest.md").read_text()
    assert "[HIGH  ] NE @ BUF   BetMGM  spreads NE +7 (-110)  Josh Allen (BUF QB) Questionable -> Out" in latest
