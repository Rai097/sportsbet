from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from sportsbet import tracking
from sportsbet.engine import scan
from sportsbet.pricing import american_to_decimal, expected_value, fair_prob_from_two_way, implied_prob
from sportsbet.providers import nflverse
from sportsbet.store import Store

KICK = datetime(2025, 10, 5, 17, 0, tzinfo=timezone.utc)
OPEN = KICK - timedelta(days=2)  # early pull, bets placed right after it
CLOSE = KICK - timedelta(hours=1)  # last pull before kickoff
LIVE = KICK + timedelta(hours=1)  # in-play pull that must never count as the close


def _rows(fetched_at, quotes):
    return [
        {
            "fetched_at": fetched_at,
            "event_id": "e1",
            "commence_time": KICK,
            "home_team": "BUF",
            "away_team": "NE",
            "bookmaker": book,
            "market": market,
            "outcome": outcome,
            "point": point,
            "price": price,
            "last_update": fetched_at,
        }
        for book, market, outcome, point, price in quotes
    ]


OPEN_QUOTES = [
    ("pinnacle", "h2h", "BUF", None, -300), ("pinnacle", "h2h", "NE", None, 250),
    ("betmgm", "h2h", "BUF", None, -320), ("betmgm", "h2h", "NE", None, 280),
    ("pinnacle", "spreads", "BUF", -7.0, -105), ("pinnacle", "spreads", "NE", 7.0, -105),
    ("betmgm", "spreads", "BUF", -7.0, -110), ("betmgm", "spreads", "NE", 7.0, -110),
    ("williamhill_us", "spreads", "BUF", -7.5, -105), ("williamhill_us", "spreads", "NE", 7.5, -115),
    ("pinnacle", "totals", "Over", 44.5, -108), ("pinnacle", "totals", "Under", 44.5, -108),
    ("betmgm", "totals", "Over", 44.5, -105), ("betmgm", "totals", "Under", 44.5, -115),
]
CLOSE_QUOTES = [
    ("pinnacle", "h2h", "BUF", None, -350), ("pinnacle", "h2h", "NE", None, 290),
    ("betmgm", "h2h", "BUF", None, -360), ("betmgm", "h2h", "NE", None, 300),
    ("pinnacle", "spreads", "BUF", -7.0, -120), ("pinnacle", "spreads", "NE", 7.0, 100),
    ("betmgm", "spreads", "BUF", -7.0, -120), ("betmgm", "spreads", "NE", 7.0, 100),
    ("williamhill_us", "spreads", "BUF", -8.0, -110), ("williamhill_us", "spreads", "NE", 8.0, -110),
    ("pinnacle", "totals", "Over", 44.5, -125), ("pinnacle", "totals", "Under", 44.5, 105),
    ("betmgm", "totals", "Over", 44.5, -125), ("betmgm", "totals", "Under", 44.5, 105),
]
LIVE_QUOTES = [
    ("pinnacle", "h2h", "BUF", None, -1000), ("pinnacle", "h2h", "NE", None, 600),
    ("betmgm", "spreads", "BUF", -7.0, -200), ("betmgm", "spreads", "NE", 7.0, 160),
]

# BUF 27, NE 20: BUF -7 pushes, NE ML loses, Over 44.5 wins, BUF -7.5 loses
SCHEDULE = pd.DataFrame(
    {
        "game_id": ["2025_05_NE_BUF", "2025_05_KC_LV"],
        "season": [2025, 2025],
        "gameday": pd.to_datetime(["2025-10-05", "2025-10-05"]),
        "home_team": ["BUF", "LV"],
        "away_team": ["NE", "KC"],
        "home_score": [27.0, None],
        "away_score": [20.0, None],
        "result": [7.0, None],
        "total": [47.0, None],
    }
)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.duckdb")
    s.insert_rows("odds_snapshots", _rows(OPEN, OPEN_QUOTES) + _rows(CLOSE, CLOSE_QUOTES) + _rows(LIVE, LIVE_QUOTES))
    # a scan of the opening pull, so the spread bet can be logged straight from a candidate
    opening = s.query("SELECT * FROM odds_snapshots WHERE fetched_at = ?", [OPEN])
    cands = [c for c in scan(opening, min_ev=-1.0) if c.market == "spreads" and c.bookmaker == "betmgm"]
    s.insert_rows("candidates", cands)
    yield s
    s.close()


@pytest.fixture
def placed(store):
    at = OPEN + timedelta(minutes=5)
    return {
        "spread": tracking.add_bet(store, matchup="NE @ BUF", bookmaker="mgm", market="spread",
                                   outcome="buf", point=-7, stake=1, placed_at=at),
        "ml": tracking.add_bet(store, matchup="NE @ BUF", bookmaker="betmgm", market="h2h",
                               outcome="NE", price=280, stake=2, placed_at=at),
        "total": tracking.add_bet(store, matchup="NE @ BUF", bookmaker="BetMGM", market="totals",
                                  outcome="over", point=44.5, price=-105, stake=1, placed_at=at),
        "moved": tracking.add_bet(store, matchup="NE @ BUF", bookmaker="caesars", market="spreads",
                                  outcome="BUF", point=-7.5, price=-105, stake=1, placed_at=at),
    }


def _bets(store):
    return store.bets_with_meta().set_index("bet_id")


def test_add_bet_from_candidate_and_manual(store, placed):
    b = _bets(store)
    spread = b.loc[placed["spread"].bet_id]
    assert spread["event_id"] == "e1" and spread["matchup"] == "NE @ BUF"
    assert spread["price"] == -110 and spread["source"] == "scan" and spread["fair_source"] == "pinnacle"
    assert spread["fair_prob"] == pytest.approx(0.5)

    ml = b.loc[placed["ml"].bet_id]
    assert ml["source"] == "manual" and pd.isna(ml["point"])
    assert ml["fair_prob"] == pytest.approx(fair_prob_from_two_way(-300, 250)[1])
    assert b.loc[placed["total"].bet_id, "fair_prob"] == pytest.approx(0.5)
    # nothing references BUF -7.5, so no fair prob at the time of the bet
    assert pd.isna(b.loc[placed["moved"].bet_id, "fair_prob"])
    assert len(store.open_bets()) == 4
    assert len({p.bet_id for p in placed.values()}) == 4


def test_add_bet_validation(store):
    with pytest.raises(ValueError, match="price"):
        # no scan candidate for this book, so the price must be supplied
        tracking.add_bet(store, matchup="NE @ BUF", bookmaker="betrivers", market="h2h", outcome="NE")
    with pytest.raises(ValueError, match="point"):
        tracking.add_bet(store, matchup="NE @ BUF", bookmaker="betmgm", market="spreads", outcome="NE", price=-110)
    with pytest.raises(ValueError, match="not in"):
        tracking.add_bet(store, matchup="NE @ BUF", bookmaker="betmgm", market="h2h", outcome="KC", price=100)
    with pytest.raises(ValueError, match="no stored odds"):
        tracking.add_bet(store, matchup="KC @ LV", bookmaker="betmgm", market="h2h", outcome="KC", price=100)
    # a game with no stored odds can still be logged when the kickoff is given
    b = tracking.add_bet(store, matchup="KC @ LV", bookmaker="betmgm", market="h2h", outcome="KC",
                         price=-150, commence_time=KICK)
    assert b.event_id is None and b.fair_prob is None


def test_capture_closing(store, placed):
    # before kickoff nothing is captured
    assert tracking.capture_closing(store, now=KICK - timedelta(minutes=1)) == 0
    assert tracking.capture_closing(store, now=KICK + timedelta(hours=2)) == 4
    b = _bets(store)

    spread = b.loc[placed["spread"].bet_id]
    assert spread["closing_price"] == -120  # the in-play -200 must be ignored
    assert spread["closing_point"] == -7
    assert spread["closing_fair_prob"] == pytest.approx(fair_prob_from_two_way(-120, 100)[0])
    assert spread["closing_fair_source"] == "pinnacle"
    assert pd.isna(spread["note"])
    assert tracking._utc(spread["closing_fetched_at"]) == pd.Timestamp(CLOSE)

    ml = b.loc[placed["ml"].bet_id]
    assert ml["closing_price"] == 300
    assert ml["closing_fair_prob"] == pytest.approx(fair_prob_from_two_way(-350, 290)[1])

    total = b.loc[placed["total"].bet_id]
    assert total["closing_price"] == -125
    assert total["closing_fair_prob"] == pytest.approx(fair_prob_from_two_way(-125, 105)[0])

    moved = b.loc[placed["moved"].bet_id]
    assert moved["closing_price"] == -110 and moved["closing_point"] == -8
    assert pd.isna(moved["closing_fair_prob"])
    assert "book closed at -8" in moved["note"] and "no reference line at -7.5" in moved["note"]

    # captured bets are not re-captured
    assert tracking.capture_closing(store, now=KICK + timedelta(hours=3)) == 0


def test_settle_with_monkeypatched_schedule(store, placed, monkeypatch):
    calls = []

    def fake_load(seasons):
        calls.append(seasons)
        return SCHEDULE.copy()

    monkeypatch.setattr(nflverse, "load_schedules", fake_load)
    assert tracking.settle(store, now=KICK - timedelta(hours=1)) == []  # not kicked off yet
    assert calls == []  # and no data loaded for nothing
    tracking.capture_closing(store, now=KICK + timedelta(hours=4))
    settled = tracking.settle(store, now=KICK + timedelta(hours=4))
    assert calls == [[2025]]
    assert sorted(settled) == sorted(p.bet_id for p in placed.values())

    b = _bets(store)
    assert b.loc[placed["spread"].bet_id, ["result", "pnl"]].tolist() == ["push", 0.0]
    assert b.loc[placed["ml"].bet_id, ["result", "pnl"]].tolist() == ["loss", -2.0]
    assert b.loc[placed["total"].bet_id, "result"] == "win"
    assert b.loc[placed["total"].bet_id, "pnl"] == pytest.approx(100 / 105)
    assert b.loc[placed["moved"].bet_id, ["result", "pnl"]].tolist() == ["loss", -1.0]
    assert store.open_bets().empty

    report = tracking.clv_report(store).set_index("group")
    overall = report.loc["overall"]
    assert overall["bets"] == 4 and overall["settled"] == 4
    assert overall["stake"] == 5
    assert overall["pnl"] == pytest.approx(-3 + 100 / 105)
    assert overall["roi"] == pytest.approx((-3 + 100 / 105) / 5)
    assert overall["clv_n"] == 3

    p_spread = fair_prob_from_two_way(-120, 100)[0]
    p_ml = fair_prob_from_two_way(-350, 290)[1]
    p_over = fair_prob_from_two_way(-125, 105)[0]
    clv = [p_spread - 0.5, p_ml - fair_prob_from_two_way(-300, 250)[1], p_over - 0.5]
    assert overall["clv_pp"] == pytest.approx(100 * sum(clv) / 3)
    beats = [implied_prob(-110) < p_spread, implied_prob(280) < p_ml, implied_prob(-105) < p_over]
    assert overall["beat_close"] == pytest.approx(sum(beats) / 3)
    evs = [expected_value(p_spread, -110), expected_value(p_ml, 280), expected_value(p_over, -105)]
    assert overall["ev_close"] == pytest.approx(sum(evs) / 3)
    cents = [
        tracking.cents(-110) - tracking.cents(tracking.american_exact(p_spread)),
        tracking.cents(280) - tracking.cents(tracking.american_exact(p_ml)),
        tracking.cents(-105) - tracking.cents(tracking.american_exact(p_over)),
    ]
    assert overall["clv_cents"] == pytest.approx(sum(cents) / 3)

    assert report.loc["Caesars", "bets"] == 1 and report.loc["Caesars", "clv_n"] == 0
    assert report.loc["BetMGM", "bets"] == 3
    assert report.loc["h2h", "pnl"] == -2.0

    text = tracking.format_report(tracking.clv_report(store))
    assert "overall" in text and "n < 100" in text
    listing = tracking.format_bets(store.bets_with_meta())
    assert "push" in listing and "Caesars" in listing


def test_settle_skips_unfinished_games(store):
    tracking.add_bet(store, matchup="KC @ LV", bookmaker="betmgm", market="h2h", outcome="KC",
                     price=-150, commence_time=KICK)
    assert tracking.settle(store, now=KICK + timedelta(hours=4), schedule=SCHEDULE) == []
    assert len(store.open_bets()) == 1


def test_settle_matches_swapped_home_team(store):
    # a neutral-site game where the schedule's designated home team differs from ours
    tracking.add_bet(store, matchup="BUF @ NE", bookmaker="betmgm", market="spreads", outcome="BUF",
                     point=-3, price=-110, commence_time=KICK)
    settled = tracking.settle(store, now=KICK + timedelta(hours=4), schedule=SCHEDULE)
    assert len(settled) == 1
    assert store.bets_with_meta().iloc[0]["result"] == "win"  # BUF by 7 covers -3


@pytest.mark.parametrize(
    "market,outcome,point,margin,total,expected",
    [
        ("h2h", "BUF", None, 3, 40, "win"),
        ("h2h", "NE", None, 3, 40, "loss"),
        ("h2h", "NE", None, 0, 40, "push"),  # tie
        ("spreads", "BUF", -3.0, 3, 40, "push"),
        ("spreads", "NE", 3.5, 3, 40, "win"),
        ("spreads", "BUF", -3.5, 3, 40, "loss"),
        ("totals", "Over", 40.0, 3, 40, "push"),
        ("totals", "Under", 40.5, 3, 40, "win"),
        ("totals", "Over", 40.5, 3, 40, "loss"),
    ],
)
def test_grade(market, outcome, point, margin, total, expected):
    assert tracking.grade(market, outcome, point, "BUF", "NE", margin, total) == expected


def test_pnl_and_helpers():
    assert tracking.bet_pnl("win", 150, 2) == pytest.approx(3.0)
    assert tracking.bet_pnl("win", -110, 1.1) == pytest.approx(1.1 * (american_to_decimal(-110) - 1))
    assert tracking.bet_pnl("loss", -110, 1.1) == -1.1
    assert tracking.bet_pnl("push", -110, 1.1) == 0.0
    assert tracking.cents(-110) == -10 and tracking.cents(120) == 20 and tracking.cents(100) == tracking.cents(-100)
    assert tracking.american_exact(0.5) == pytest.approx(100)
    assert tracking.american_exact(0.6) == pytest.approx(-150)
    lo, hi = tracking.wilson_interval(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60
    assert tracking.parse_matchup("ne @ Buffalo Bills") == ("NE", "BUF")


def test_update_bet_routes_fields(store, placed):
    bid = placed["ml"].bet_id
    store.update_bet(bid, closing_price=310.0, note="hand edit")
    row = _bets(store).loc[bid]
    assert row["closing_price"] == 310 and row["note"] == "hand edit"
    with pytest.raises(ValueError):
        store.update_bet(bid, nonsense=1)
    assert store.last_snapshot_before("e1", KICK)["fetched_at"].map(tracking._utc).unique().tolist() == [
        pd.Timestamp(CLOSE)
    ]
