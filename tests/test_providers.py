import json
from pathlib import Path

import pytest
import requests

from sportsbet.config import Settings
from sportsbet.providers.espn import parse_espn_injuries
from sportsbet.providers.odds_api import OddsApiClient, normalize

FIX = Path(__file__).parent / "fixtures"


def test_normalize_odds_api_fixture():
    raw = json.loads((FIX / "odds_api_nfl_sample.json").read_text())
    rows = normalize(raw)
    assert len(rows) == 6 * 3 + 2 * 3  # evt1: 3 books x 3 markets x 2; evt2: 3 books x 1 market x 2
    r = rows[0]
    assert r.home_team == "BUF" and r.away_team == "NE"
    assert r.bookmaker == "pinnacle" and r.market == "h2h"
    assert r.outcome in {"BUF", "NE"}
    spreads = [x for x in rows if x.market == "spreads" and x.bookmaker == "williamhill_us"]
    assert {x.point for x in spreads} == {-7.5, 7.5}


def test_parse_espn_injuries_fixture():
    payload = json.loads((FIX / "espn_injuries_sample.json").read_text())
    rows = parse_espn_injuries(payload)
    assert len(rows) == 3
    allen = next(r for r in rows if r.player == "Josh Allen")
    assert allen.team == "BUF" and allen.position == "QB" and allen.status == "Out"
    ir = next(r for r in rows if r.team == "NE")
    assert ir.status == "Out"  # IR normalised to Out


class _Session:
    """Fails like requests does: the message carries the full URL, query string included."""

    def __init__(self, exc_type, status=None):
        self.exc_type, self.status = exc_type, status

    def get(self, url, params=None, timeout=None):
        full = requests.Request("GET", url, params=params).prepare().url
        if self.status is None:
            raise self.exc_type(f"Max retries exceeded with url: {full}")
        resp = requests.Response()
        resp.status_code, resp.url, resp.reason = self.status, full, "Unauthorized"
        return resp


@pytest.mark.parametrize("session", [_Session(requests.ConnectionError), _Session(None, status=401)])
def test_odds_api_errors_do_not_leak_the_key(tmp_path, session):
    settings = Settings(odds_api_key="sekret123", data_dir=tmp_path, db_path=tmp_path / "t.duckdb")
    client = OddsApiClient(settings, session=session)
    with pytest.raises(requests.RequestException) as info:
        client.fetch_nfl_odds()
    assert "sekret123" not in str(info.value)
    assert "<ODDS_API_KEY>" in str(info.value)
