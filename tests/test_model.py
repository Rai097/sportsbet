import pandas as pd
import pytest

from sportsbet.model.elo import EloModel
from sportsbet.model.injuries import matchup_adjustment, starters_from_depth_chart, team_impacts


def test_elo_home_field_and_updates():
    m = EloModel()
    pred = m.predict("A", "B")
    assert pred["home_spread"] == 2.5
    assert 0.5 < pred["home_win_prob"] < 0.6
    m.update("A", "B", 30, 10)
    assert m.rating("A") > m.rating("B")
    assert m.predict("A", "B")["home_win_prob"] > pred["home_win_prob"]


def test_elo_fit_regresses_between_seasons():
    games = pd.DataFrame(
        {
            "season": [2020, 2020, 2021],
            "gameday": pd.to_datetime(["2020-09-10", "2020-09-17", "2021-09-09"]),
            "gametime": ["20:00"] * 3,
            "home_team": ["A", "A", "A"],
            "away_team": ["B", "B", "B"],
            "home_score": [40, 40, 40],
            "away_score": [0, 0, 0],
            "result": [40, 40, 40],
            "location": ["Home"] * 3,
        }
    )
    m1 = EloModel().fit(games.iloc[:2])
    m2 = EloModel().fit(games)
    # After the season boundary A was regressed toward the mean before the third game moved it again,
    # so the gain from game 3 is less than the sum of gains from games 1 and 2.
    assert m2.rating("A") - 1505 < 1.5 * (m1.rating("A") - 1505)


def test_injury_impacts_and_adjustment():
    inj = pd.DataFrame(
        {
            "team": ["BUF", "BUF", "NE"],
            "player": ["Josh Allen", "Some Backup", "Starting Tackle"],
            "position": ["QB", "WR", "OT"],
            "status": ["Out", "Questionable", "Out"],
        }
    )
    starters = {("BUF", "Josh Allen"), ("NE", "Starting Tackle")}
    imp = team_impacts(inj, starters)
    assert imp["BUF"].points == 3.8  # backup WR questionable is below the 0.05 threshold
    assert imp["NE"].points == 0.6
    # BUF hosting NE: BUF loses 3.8, NE loses 0.6 -> adjustment favours away by 3.2
    assert matchup_adjustment(imp, "BUF", "NE") == pytest.approx(-3.2)
    assert matchup_adjustment(imp, "NE", "BUF") == pytest.approx(3.2)


def test_starters_from_current_depth_chart_schema():
    depth = pd.DataFrame(
        {
            "dt": ["2026-09-22T00:00:00Z", "2026-09-29T00:00:00Z", "2026-09-29T00:00:00Z"],
            "team": ["BUF", "BUF", "BUF"],
            "player_name": ["Old Starter", "Josh Allen", "Backup QB"],
            "pos_rank": [1, 1, 2],
        }
    )
    assert starters_from_depth_chart(depth) == {("BUF", "Josh Allen")}
