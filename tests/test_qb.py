import pandas as pd
import pytest

from sportsbet import backtest as bt
from sportsbet.model.elo import EloModel
from sportsbet.model.injuries import QB_POINTS_PER_VALUE, team_impacts
from sportsbet.model.qb import (
    QbConfig,
    QbEloModel,
    QbRatings,
    measure_market_qb_price,
    qb_game_values,
    qb_injury_gaps,
)


def _schedule(n_weeks: int = 6, backup_week: int | None = None) -> pd.DataFrame:
    """A plays B every week at home; A's starter is 'a1' unless backup_week says 'a2'."""
    rows = []
    for w in range(1, n_weeks + 1):
        rows.append(
            {
                "game_id": f"2020_{w:02d}_B_A",
                "season": 2020,
                "week": w,
                "game_type": "REG",
                "gameday": pd.Timestamp("2020-09-10") + pd.Timedelta(days=7 * (w - 1)),
                "gametime": "13:00",
                "home_team": "A",
                "away_team": "B",
                "home_score": 24 if w % 2 else 17,
                "away_score": 20,
                "result": (24 if w % 2 else 17) - 20,
                "location": "Home",
                "home_moneyline": -150,
                "away_moneyline": 130,
                "spread_line": 3.0,
                "home_qb_id": "a2" if w == backup_week else "a1",
                "away_qb_id": "b1",
                "home_qb_name": "Backup A" if w == backup_week else "Star A",
                "away_qb_name": "Starter B",
            }
        )
    return pd.DataFrame(rows)


def _game_values(sched: pd.DataFrame, star: float = 12.0, other: float = 0.0) -> pd.DataFrame:
    rows = []
    for g in sched.itertuples(index=False):
        rows.append({"game_id": g.game_id, "team": "A", "qb_id": g.home_qb_id, "qb_name": g.home_qb_name,
                     "value": star if g.home_qb_id == "a1" else other, "weight": 1.0})
        rows.append({"game_id": g.game_id, "team": "B", "qb_id": "b1", "qb_name": "Starter B", "value": other, "weight": 1.0})
    return pd.DataFrame(rows)


def test_qb_value_rises_after_good_games():
    r = QbRatings(config=QbConfig(opp_adjust=False))
    start = r.value("q")
    assert start == r.config.replacement  # unknown QB is replacement level
    r.observe("A", "B", "q", [("q", "Q", 15.0, 1.0)])
    after_one = r.value("q")
    r.observe("A", "B", "q", [("q", "Q", 15.0, 1.0)])
    assert start < after_one < r.value("q") < 15.0
    # a bad game pulls him back down
    before = r.value("q")
    r.observe("A", "B", "q", [("q", "Q", -20.0, 1.0)])
    assert r.value("q") < before
    # a two-snap cameo moves him far less than a full start
    r2 = QbRatings(config=QbConfig(opp_adjust=False))
    r2.observe("A", "B", "q", [("q", "Q", 15.0, 0.05)])
    assert r2.value("q") - start < 0.1 * (after_one - start)


def test_qb_swap_moves_prediction_by_value_gap():
    sched = _schedule(n_weeks=5)
    m = QbEloModel.from_game_values(_game_values(sched), QbConfig(opp_adjust=False)).fit(sched)
    nxt = _schedule(n_weeks=6, backup_week=6).iloc[[-1]].copy()
    nxt["result"] = float("nan")
    g_backup = next(nxt.itertuples(index=False))
    g_star = next(nxt.assign(home_qb_id="a1", home_qb_name="Star A").itertuples(index=False))
    gap = m.qb.value("a1") - m.qb.value("a2")
    assert gap > 0
    diff = m.predict_game(g_star)["home_spread"] - m.predict_game(g_backup)["home_spread"]
    assert diff == pytest.approx(m.qb.config.points_per_value * gap)
    # The usual starter sits close to the team baseline, so he barely moves the Elo line.
    plain = EloModel().fit(sched).predict_game(g_star)["home_spread"]
    assert abs(m.predict_game(g_star)["home_spread"] - plain) < 0.5 * m.qb.config.points_per_value * gap


def test_week_falls_back_to_name_then_last_starter():
    sched = _schedule(n_weeks=5)
    m = QbEloModel.from_game_values(_game_values(sched), QbConfig(opp_adjust=False)).fit(sched)
    row = sched.iloc[[0]].drop(columns=["home_qb_id", "away_qb_id"]).assign(home_qb_name="Star A", away_qb_name=None)
    assert m.starters(next(row.itertuples(index=False))) == ("a1", "b1")


def test_plain_elo_unchanged_by_hooks():
    sched = _schedule(n_weeks=6)
    a = EloModel().fit(sched)
    b = EloModel()
    for g in sched.itertuples(index=False):
        b.update(g.home_team, g.away_team, g.home_score, g.away_score)
    assert a.ratings == b.ratings


def test_backtest_reports_both_models():
    sched = _schedule(n_weeks=8, backup_week=7)
    res = bt.evaluate(sched, eval_from=2020, qb_values=_game_values(sched), qb_config=QbConfig(opp_adjust=False))
    m = res.metrics
    for key in ("brier_model", "brier_qb", "logloss_model", "logloss_qb", "brier_market", "ml_roi", "qb_ml_roi", "ats_roi", "qb_ats_roi"):
        assert key in m
    assert m["games"] == 8
    assert m["brier_qb"] != m["brier_model"]  # the backup game changed the QB-aware forecast
    plain_only = bt.evaluate(sched, eval_from=2020)
    assert plain_only.metrics["brier_model"] == m["brier_model"]
    assert "brier_qb" not in plain_only.metrics
    table = bt.comparison_table(m)
    assert list(table.columns) == ["elo", "qb_elo", "market"]
    if not res.bets.empty:
        assert set(res.bets["model"]) <= {"elo", "qb"}


def test_qb_game_values_from_player_stats():
    ps = pd.DataFrame(
        {
            "player_id": ["q1", "q2", "wr"],
            "player_display_name": ["Q One", "Q Two", "Wide"],
            "position": ["QB", "QB", "WR"],
            "game_id": ["g"] * 3,
            "team": ["A"] * 3,
            "attempts": [34, 2, 0],
            "sacks_suffered": [2, 0, 0],
            "carries": [2, 0, 5],
            "passing_epa": [7.6, -1.0, 0.0],
            "rushing_epa": [0.0, None, 2.0],
        }
    )
    gv = qb_game_values(ps).set_index("qb_id")
    assert set(gv.index) == {"q1", "q2"}
    assert gv.loc["q1", "value"] == pytest.approx(7.6)  # 38 plays: per-play EPA x 38
    assert gv.loc["q1", "weight"] == 1.0
    assert gv.loc["q2", "weight"] == pytest.approx(2 / 38)


def test_market_price_detects_starter_to_backup():
    sched = _schedule(n_weeks=6, backup_week=6)
    # Make the market agree with plain Elo except in the backup game, where it moves 3 points.
    wf = bt.walk_forward(sched, 2020).set_index("game_id")["home_spread"]
    sched["spread_line"] = sched["game_id"].map(wf)
    sched.loc[sched["week"] == 6, "spread_line"] -= 3.0
    price = measure_market_qb_price(sched, _game_values(sched), QbConfig(opp_adjust=False))
    assert price.n_backup == 1
    assert price.backup_shift == pytest.approx(3.0)
    assert price.backup_value_gap > 0


def test_injury_uses_value_gap_when_known():
    sched = _schedule(n_weeks=5)
    m = QbEloModel.from_game_values(_game_values(sched), QbConfig(opp_adjust=False)).fit(sched)
    depth = pd.DataFrame(
        {
            "dt": ["2020-10-10"] * 3,
            "team": ["A", "A", "A"],
            "player_name": ["Star A", "Backup A", "Wide"],
            "gsis_id": ["a1", "a2", "w1"],
            "pos_abb": ["QB", "QB", "WR"],
            "pos_rank": [1, 2, 1],
        }
    )
    gaps = qb_injury_gaps(m, depth)
    assert set(gaps) == {("A", "Star A")}
    inj = pd.DataFrame({"team": ["A"], "player": ["Star A"], "position": ["QB"], "status": ["Out"]})
    flat = team_impacts(inj, {("A", "Star A")})["A"].points
    priced = team_impacts(inj, {("A", "Star A")}, qb_gaps=gaps)["A"].points
    assert flat == 3.8
    assert priced == pytest.approx(min(gaps[("A", "Star A")] * QB_POINTS_PER_VALUE, 10.0))
    # A backup as good as the starter costs nothing.
    assert "A" not in team_impacts(inj, None, qb_gaps={("A", "Star A"): -1.0})
