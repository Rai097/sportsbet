import json
from pathlib import Path

import pandas as pd
import pytest

from sportsbet import pushchart
from sportsbet.engine import scan
from sportsbet.providers.odds_api import normalize
from sportsbet.pricing import expected_value, fair_prob_from_two_way
from test_pushchart import synthetic_chart

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _offline_push_chart(monkeypatch):
    # scan loads the cached chart on first need; pin a synthetic one so tests never hit nflverse
    monkeypatch.setattr(pushchart, "_default", synthetic_chart())


def _odds_df():
    raw = json.loads((FIX / "odds_api_nfl_sample.json").read_text())
    return pd.DataFrame([r.as_dict() for r in normalize(raw)])


def _find(cands, book, market, outcome):
    return next(c for c in cands if c.bookmaker == book and c.market == market and c.outcome == outcome)


def test_scan_finds_mgm_dog_vs_pinnacle():
    cands = scan(_odds_df(), min_ev=0.0)
    keys = {(c.bookmaker, c.market, c.outcome, c.point) for c in cands}
    # Pinnacle fair for NE +250 vs MGM +285 -> positive EV
    assert ("betmgm", "h2h", "NE", None) in keys
    ne = _find(cands, "betmgm", "h2h", "NE")
    _, p_ne = fair_prob_from_two_way(-290, 250)
    assert ne.fair_prob == pd.Series([p_ne]).iloc[0]
    assert ne.ev == expected_value(p_ne, 285)
    assert ne.fair_source == "pinnacle"
    # sorted by EV desc
    assert [c.ev for c in cands] == sorted([c.ev for c in cands], reverse=True)


def test_caesars_half_point_spread_is_converted_from_pinnacle():
    chart = synthetic_chart()
    cands = scan(_odds_df(), min_ev=-1.0, push_chart=chart)
    ne = _find(cands, "williamhill_us", "spreads", "NE")
    buf = _find(cands, "williamhill_us", "spreads", "BUF")
    assert (ne.point, ne.fair_source) == (7.5, "pinnacle@+7.0")
    assert (buf.point, buf.fair_source) == (-7.5, "pinnacle@-7.0")
    assert ne.fair_prob + buf.fair_prob == pytest.approx(1.0)
    # the conversion goes through the chart exactly: Pinnacle -7 -> fair line -> +7.5
    p_buf, _ = fair_prob_from_two_way(-104, -106)
    fair_line = chart.fair_spread_from_prob(-7.0, p_buf)
    assert ne.fair_prob == pytest.approx(chart.cover_prob(fair_line, 7.5, "away"))
    # NE +7.5 wins every push at 7 that Pinnacle's NE +7 refunded, so it must be worth more
    win7, push7, loss7 = chart.spread_probs(fair_line, 7.0, "away")
    assert ne.fair_prob == pytest.approx(win7 + push7)
    assert ne.fair_prob > 1 - p_buf
    assert ne.ev == pytest.approx(expected_value(ne.fair_prob, -115))
    # same-point prices keep the plain de-vig source
    assert _find(cands, "betmgm", "spreads", "NE").fair_source == "pinnacle"


def test_converted_candidate_clears_min_ev_when_key_number_is_heavy():
    chart = synthetic_chart()
    chart.margin.weights[chart.margin.support == 7] *= 1.6
    chart.margin.weights[chart.margin.support == -7] *= 1.6
    cands = scan(_odds_df(), min_ev=0.0, push_chart=chart)
    ne = _find(cands, "williamhill_us", "spreads", "NE")
    assert ne.fair_source == "pinnacle@+7.0" and ne.ev > 0


def test_pinnacle_alternate_at_target_point_beats_conversion():
    df = _odds_df()
    pin = df[(df.bookmaker == "pinnacle") & (df.market == "spreads")].copy()
    pin["point"] = pin["point"] * 7.5 / 7.0
    pin["price"] = [110.0, -130.0]
    cands = scan(pd.concat([df, pin], ignore_index=True), min_ev=-1.0)
    ne = _find(cands, "williamhill_us", "spreads", "NE")
    assert ne.fair_source == "pinnacle"
    _, p_ne = fair_prob_from_two_way(110, -130)
    assert ne.fair_prob == pytest.approx(p_ne)


def test_totals_converted_across_points():
    df = _odds_df()
    mask = (df.bookmaker == "williamhill_us") & (df.market == "totals")
    df.loc[mask, "point"] = 48.5
    cands = scan(df, min_ev=-1.0)
    over = _find(cands, "williamhill_us", "totals", "Over")
    under = _find(cands, "williamhill_us", "totals", "Under")
    assert over.fair_source == "pinnacle@47.5"
    p_over, _ = fair_prob_from_two_way(-103, -107)
    assert over.fair_prob < p_over  # a higher total is harder to go over
    assert over.fair_prob + under.fair_prob == pytest.approx(1.0)


def test_consensus_line_used_for_conversion_without_pinnacle():
    df = _odds_df()
    df = df[~((df.bookmaker == "pinnacle") & (df.market == "spreads"))]
    base = _odds_df()
    base = base[(base.bookmaker == "pinnacle") & (base.market == "spreads")]
    refs = pd.concat([base.assign(bookmaker="draftkings"), base.assign(bookmaker="fanduel")])
    cands = scan(pd.concat([df, refs], ignore_index=True), min_ev=-1.0)
    ne = _find(cands, "williamhill_us", "spreads", "NE")
    assert ne.fair_source == "consensus(2)@+7.0"


def test_flipped_sign_is_not_treated_as_same_point():
    df = _odds_df()
    mask = (df.bookmaker == "betmgm") & (df.market == "spreads")
    df.loc[mask, "point"] = -df.loc[mask, "point"]  # BUF +7 / NE -7 shares abs(point) with Pinnacle
    cands = scan(df, min_ev=-1.0)
    buf = _find(cands, "betmgm", "spreads", "BUF")
    assert buf.point == 7.0 and buf.fair_source == "pinnacle@-7.0"
    assert buf.fair_prob > 0.75


def test_scan_without_push_chart_skips_conversion(monkeypatch):
    monkeypatch.setattr(pushchart, "_default", None)

    def offline():
        raise OSError("no network")

    monkeypatch.setattr(pushchart, "load", lambda *a, **k: offline())
    cands = scan(_odds_df(), min_ev=-1.0)
    assert not any(c.bookmaker == "williamhill_us" and c.market == "spreads" for c in cands)
    assert any(c.bookmaker == "betmgm" and c.market == "spreads" for c in cands)


def test_scan_consensus_fallback_when_no_pinnacle():
    cands = scan(_odds_df(), min_ev=0.0)
    lv = [c for c in cands if c.event_id == "evt2"]
    assert lv, "expected a consensus-based candidate on the KC/LV game"
    assert all(c.fair_source == "consensus(2)" for c in lv)
    assert any(c.outcome == "LV" and c.price == 195 for c in lv)


def test_min_ev_filters():
    assert len(scan(_odds_df(), min_ev=0.0)) > len(scan(_odds_df(), min_ev=0.05))


def test_model_probs_attached_to_h2h():
    df = _odds_df()
    cands = scan(df, min_ev=0.0, model_probs={("evt1", "NE"): 0.3})
    ne = _find(cands, "betmgm", "h2h", "NE")
    assert ne.model_prob == 0.3


def test_latest_odds_drops_quotes_from_older_pulls(tmp_path):
    """A line that moved off a point, or a market taken down, must not be scanned again."""
    from datetime import datetime, timedelta, timezone

    from sportsbet.store import Store

    t0 = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
    first = _odds_df().assign(fetched_at=t0)
    later = first[first.bookmaker != "williamhill_us"].copy()  # Caesars pulled its markets
    later["fetched_at"] = t0 + timedelta(hours=2)
    pin = (later.bookmaker == "pinnacle") & (later.market == "spreads") & (later.event_id == "evt1")
    later.loc[pin, "point"] = later.loc[pin, "point"] * 4.5 / 7.0  # Pinnacle BUF -7 -> -4.5
    store = Store(tmp_path / "t.duckdb")
    store.insert_rows("odds_snapshots", first.to_dict("records"))
    store.insert_rows("odds_snapshots", later.to_dict("records"))

    latest = store.latest_odds()
    assert set(latest["fetched_at"].dt.tz_convert("UTC")) == {pd.Timestamp(t0 + timedelta(hours=2))}
    assert "williamhill_us" not in set(latest["bookmaker"])
    pin_pts = latest[(latest.bookmaker == "pinnacle") & (latest.market == "spreads")]["point"]
    assert sorted(pin_pts.abs()) == [4.5, 4.5]

    # BetMGM still dealing NE +7 against a -4.5 market is the edge, priced off the live line
    ne = _find(scan(latest, min_ev=0.0), "betmgm", "spreads", "NE")
    assert ne.fair_source == "pinnacle@+4.5" and ne.ev > 0.05
    store.close()
