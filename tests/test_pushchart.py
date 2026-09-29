import argparse

import numpy as np
import pandas as pd
import pytest

from sportsbet import pushchart as pc
from sportsbet.pricing import prob_to_american


def synthetic_chart() -> pc.PushChart:
    """Small hand-set chart: normal kernel plus spikes on the NFL key numbers."""
    m_support = np.arange(-60, 61, dtype=float)
    m_w = np.ones(len(m_support))
    for key, mult in {0: 0.15, 1: 0.9, 2: 0.9, 3: 3.0, 4: 1.1, 6: 1.6, 7: 2.2, 10: 1.6, 14: 1.6}.items():
        m_w[m_support == key] = mult
        m_w[m_support == -key] = mult
    t_support = np.arange(0, 101, dtype=float)
    t_w = np.ones(len(t_support))
    for key in (37, 41, 44, 47, 51):
        t_w[t_support == key] = 1.4
    return pc.PushChart(pc.Dist(m_support, m_w, 13.0), pc.Dist(t_support, t_w, 13.0), 0, (0, 0))


@pytest.fixture
def chart():
    return synthetic_chart()


def test_three_is_the_most_common_margin(chart):
    pmf = chart.margin_pmf(-3.0)
    assert pmf.idxmax() == 3
    assert pmf[3] > 1.5 * max(pmf[2], pmf[4])
    # a pick'em game still lands on +-3 more than anything else
    pk = chart.margin_pmf(0.0)
    by_abs = pk.groupby(pk.index.map(abs)).sum()
    assert by_abs.idxmax() == 3
    assert pmf.sum() == pytest.approx(1.0)


def test_fair_line_is_the_fifty_percent_point(chart):
    for fair in (-7.0, -3.0, -1.0, 2.5, 6.5):
        # home laying the fair number is a coin flip once pushes are removed
        assert chart.cover_prob(fair, fair, "home") == pytest.approx(0.5, abs=1e-6)
        assert chart.cover_prob(fair, -fair, "away") == pytest.approx(0.5, abs=1e-6)


def test_cover_prob_monotone_in_point_and_line(chart):
    points = np.arange(-14.0, 14.01, 0.5)
    home = [chart.cover_prob(-3.0, p, "home") for p in points]
    assert all(b > a for a, b in zip(home, home[1:]))  # more points, better cover
    away = [chart.cover_prob(-3.0, p, "away") for p in points]
    assert all(b > a for a, b in zip(away, away[1:]))
    lines = np.arange(-10.0, 10.01, 0.25)
    by_line = [chart.cover_prob(f, -3.0, "home") for f in lines]
    assert all(b < a for a, b in zip(by_line, by_line[1:]))  # weaker home team, worse cover


def test_home_and_away_are_complements(chart):
    for pt in (-7.0, -6.5, -3.0, 2.5):
        win, push, loss = chart.spread_probs(-4.0, pt, "home")
        a_win, a_push, a_loss = chart.spread_probs(-4.0, -pt, "away")
        assert (win, push, loss) == pytest.approx((a_loss, a_push, a_win))
        assert win + push + loss == pytest.approx(1.0)


def test_half_point_through_key_number_is_worth_the_push(chart):
    win_3, push_3, _ = chart.spread_probs(-3.0, -3.0, "home")
    win_25, push_25, _ = chart.spread_probs(-3.0, -2.5, "home")
    assert push_25 == 0.0
    assert win_25 - win_3 == pytest.approx(push_3)
    # buying on to 3 is worth more than buying on to 4
    _, push_4, _ = chart.spread_probs(-3.0, -4.0, "home")
    assert push_3 > 2 * push_4


def _vigged(p: float) -> tuple[float, float]:
    return prob_to_american(p**VIG_EXP), prob_to_american((1 - p) ** VIG_EXP)


VIG_EXP = 0.94  # about a 4% hold on an even market


@pytest.mark.parametrize("fair", [-10.5, -7.2, -6.8, -3.0, -2.5, 0.0, 1.5, 4.0])
@pytest.mark.parametrize("point", [-7.0, -3.5, -3.0, 0.0, 2.5])
def test_fair_spread_roundtrip(chart, fair, point):
    p_home = chart.cover_prob(fair, point, "home")
    assert chart.fair_spread_from_prob(point, p_home) == pytest.approx(fair, abs=1e-3)
    # vig added the way the power de-vig removes it, then rounded to whole American odds
    price_home, price_away = _vigged(p_home)
    assert chart.fair_spread_from_prices(point, price_home, price_away) == pytest.approx(fair, abs=0.25)


def test_totals(chart):
    assert chart.over_prob(44.0, 44.0) == pytest.approx(0.5, abs=1e-6)
    overs = [chart.over_prob(44.5, t) for t in np.arange(38.0, 52.01, 0.5)]
    assert all(b < a for a, b in zip(overs, overs[1:]))
    over, push, under = chart.total_probs(47.0, 47.0)
    assert over + push + under == pytest.approx(1.0)
    assert push > chart.total_probs(47.0, 46.0)[1]  # 47 is weighted as a key total, 46 is not
    for fair in (38.5, 44.0, 47.3, 51.0):
        p = chart.over_prob(fair, 47.5)
        assert chart.fair_total_from_prob(47.5, p) == pytest.approx(fair, abs=1e-3)
        o, u = _vigged(p)
        assert chart.fair_total_from_prices(47.5, o, u) == pytest.approx(fair, abs=0.25)


def _synthetic_schedule(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    lines = rng.choice([-7.0, -3.0, -2.5, 0.0, 1.0, 3.0, 3.5, 6.5, 7.0, 10.0], size=n)
    margin = np.round(rng.normal(lines, 13.0)).astype(int)
    # snap a share of close results onto +-3 and +-7 the way field goals and touchdowns do
    for key in (3, 7):
        for sign in (1, -1):
            near = (np.abs(margin - sign * key) <= 1) & (rng.random(n) < 0.6)
            margin[near] = sign * key
    totals = np.round(rng.normal(45.0, 12.0, size=n)).clip(3, 100).astype(int)
    return pd.DataFrame(
        {
            "season": 2020,
            "game_type": "REG",
            "spread_line": lines,
            "result": margin,
            "total_line": 45.0,
            "total": totals,
            "home_spread_odds": -110.0,
            "away_spread_odds": -110.0,
            "over_odds": -110.0,
            "under_odds": -110.0,
        }
    )


def test_fit_recovers_key_numbers():
    chart = pc.fit(_synthetic_schedule())
    w = pd.Series(chart.margin.weights, index=chart.margin.support.astype(int))
    assert w[3] > 1.5 * max(w[2], w[4])
    assert w[7] > 1.5 * max(w[6], w[8])
    assert w[3] == pytest.approx(w[-3])  # symmetric by construction
    assert chart.n_games == 600
    assert pc.top_margins(chart)["margin"].iloc[0] == 3


def test_cache_roundtrip_and_load_without_network(tmp_path, monkeypatch, chart):
    path = tmp_path / "pushchart.json"
    chart.save(path)

    def no_build(*a, **k):
        raise AssertionError("load should not rebuild when the cache is current")

    monkeypatch.setattr(pc, "build", no_build)
    loaded = pc.load(path)
    assert loaded.cover_prob(-7.0, -7.5) == pytest.approx(chart.cover_prob(-7.0, -7.5))
    assert loaded.over_prob(44.0, 47.5) == pytest.approx(chart.over_prob(44.0, 47.5))


def test_stale_cache_triggers_rebuild(tmp_path, monkeypatch, chart):
    path = tmp_path / "pushchart.json"
    path.write_text('{"version": 0}')
    monkeypatch.setattr(pc, "build", lambda p: chart)
    assert pc.load(path) is chart


def test_cli_prints_tables(monkeypatch, capsys, chart):
    chart.margin.counts = np.zeros(len(chart.margin.support))
    chart.margin.counts[chart.margin.support == 3] = 10
    chart.margin.counts[chart.margin.support == -7] = 5
    monkeypatch.setattr(pc, "load", lambda rebuild=False: chart)
    parser = argparse.ArgumentParser()
    pc.register(parser.add_subparsers(dest="cmd"))
    args = parser.parse_args(["pushchart", "--rebuild"])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "top margins" in out and "half-point value" in out
    table = pc.half_point_table(chart)
    assert len(table) == 10
    assert table.loc[table["point"] == "-2.5", "dwin"].iloc[0] == pytest.approx(
        chart.spread_probs(-3.0, -3.0)[1], abs=1e-4
    )


@pytest.mark.network
def test_real_nflverse_push_chart(tmp_path):
    chart = pc.build(tmp_path / "pushchart.json")
    assert chart.n_games > 4000
    top = pc.top_margins(chart)
    assert list(top["margin"].iloc[:2]) == [3, 7]
    pmf = chart.margin_pmf(-3.0)
    assert pmf.idxmax() == 3
    # key numbers survive away from the key: a 5.5 point favourite still lands on 3 and 7 most
    pmf55 = chart.margin_pmf(-5.5)
    assert pmf55[3] > pmf55[4] and pmf55[7] > pmf55[8] and pmf55[7] > pmf55[5]
    # the half point through 3 is worth several times the half point through 4
    push3 = chart.spread_probs(-3.0, -3.0)[1]
    push4 = chart.spread_probs(-3.0, -4.0)[1]
    assert 0.06 < push3 < 0.11 and push3 > 2 * push4
    push7 = chart.spread_probs(-7.0, -7.0)[1]
    assert 0.04 < push7 < 0.08
    assert 11.0 <= chart.margin.sigma <= 16.0
    assert 11.0 <= chart.total.sigma <= 16.0
