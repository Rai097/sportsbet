import json
from pathlib import Path

import pandas as pd

from sportsbet.engine import scan
from sportsbet.providers.odds_api import normalize
from sportsbet.pricing import expected_value, fair_prob_from_two_way

FIX = Path(__file__).parent / "fixtures"


def _odds_df():
    raw = json.loads((FIX / "odds_api_nfl_sample.json").read_text())
    return pd.DataFrame([r.as_dict() for r in normalize(raw)])


def test_scan_finds_mgm_dog_vs_pinnacle():
    cands = scan(_odds_df(), min_ev=0.0)
    keys = {(c.bookmaker, c.market, c.outcome, c.point) for c in cands}
    # Pinnacle fair for NE +250 vs MGM +285 -> positive EV
    assert ("betmgm", "h2h", "NE", None) in keys
    ne = next(c for c in cands if c.bookmaker == "betmgm" and c.market == "h2h" and c.outcome == "NE")
    _, p_ne = fair_prob_from_two_way(-290, 250)
    assert ne.fair_prob == pd.Series([p_ne]).iloc[0]
    assert ne.ev == expected_value(p_ne, 285)
    assert ne.fair_source == "pinnacle"
    # Caesars quotes a different spread point (7.5) so it must not be compared to Pinnacle's 7.0
    assert not any(c.bookmaker == "williamhill_us" and c.market == "spreads" for c in cands)
    # sorted by EV desc
    assert [c.ev for c in cands] == sorted([c.ev for c in cands], reverse=True)


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
    ne = next(c for c in cands if c.bookmaker == "betmgm" and c.market == "h2h" and c.outcome == "NE")
    assert ne.model_prob == 0.3
