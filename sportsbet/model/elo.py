"""Margin-adjusted Elo for NFL teams.

Follows the well-known FiveThirtyEight recipe: K=20, home field worth ~2.5 points,
margin-of-victory multiplier with autocorrelation damping, and one-third regression
to the mean between seasons. 25 Elo points equal one point of spread.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

ELO_PER_POINT = 25.0
MEAN_ELO = 1505.0


@dataclass
class EloConfig:
    k: float = 20.0
    home_adv_points: float = 2.5
    season_regression: float = 1.0 / 3.0
    rest_points_per_day: float = 0.0  # left at 0: nflverse rest columns are unreliable pre-2013
    initial: float = MEAN_ELO


@dataclass
class EloModel:
    config: EloConfig = field(default_factory=EloConfig)
    ratings: dict[str, float] = field(default_factory=dict)
    last_season: int | None = None

    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.config.initial)

    def _new_season(self, season: int) -> None:
        if self.last_season is not None and season != self.last_season:
            r = self.config.season_regression
            for t in self.ratings:
                self.ratings[t] = self.ratings[t] * (1 - r) + MEAN_ELO * r
        self.last_season = season

    def pregame_diff(self, home: str, away: str, neutral: bool = False, extra_points: float = 0.0) -> float:
        """Home minus away rating difference in Elo points, including home field.

        extra_points is a spread-scale adjustment (positive favours home), e.g. from injuries.
        """
        hfa = 0.0 if neutral else self.config.home_adv_points * ELO_PER_POINT
        return self.rating(home) - self.rating(away) + hfa + extra_points * ELO_PER_POINT

    @staticmethod
    def win_prob_from_diff(diff: float) -> float:
        return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    @staticmethod
    def spread_from_diff(diff: float) -> float:
        """Predicted home margin (positive = home favoured)."""
        return diff / ELO_PER_POINT

    def predict(self, home: str, away: str, neutral: bool = False, extra_points: float = 0.0) -> dict[str, float]:
        diff = self.pregame_diff(home, away, neutral, extra_points)
        return {
            "home_win_prob": self.win_prob_from_diff(diff),
            "home_spread": self.spread_from_diff(diff),
        }

    def update(
        self, home: str, away: str, home_score: float, away_score: float, neutral: bool = False, extra_points: float = 0.0
    ) -> None:
        """extra_points shifts the pregame expectation, so a result is judged against it."""
        diff = self.pregame_diff(home, away, neutral, extra_points)
        expected = self.win_prob_from_diff(diff)
        margin = home_score - away_score
        actual = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
        # margin-of-victory multiplier with damping when the favourite covers big
        fav_diff = diff if margin >= 0 else -diff
        mov = math.log(abs(margin) + 1.0) * (2.2 / (fav_diff * 0.001 + 2.2))
        shift = self.config.k * mov * (actual - expected)
        self.ratings[home] = self.rating(home) + shift
        self.ratings[away] = self.rating(away) - shift

    # Schedule-row hooks. Subclasses (e.g. the QB-aware model) override these to use
    # per-game columns such as the listed starting quarterbacks.
    def predict_game(self, g: Any, extra_points: float = 0.0) -> dict[str, float]:
        neutral = getattr(g, "location", "Home") == "Neutral"
        return self.predict(g.home_team, g.away_team, neutral=neutral, extra_points=extra_points)

    def update_game(self, g: Any) -> None:
        neutral = getattr(g, "location", "Home") == "Neutral"
        self.update(g.home_team, g.away_team, float(g.home_score), float(g.away_score), neutral=neutral)

    def fit(self, games: pd.DataFrame) -> "EloModel":
        """Consume completed games in chronological order."""
        played = games[games["result"].notna()].sort_values(["season", "gameday", "gametime"])
        for g in played.itertuples(index=False):
            self._new_season(int(g.season))
            self.update_game(g)
        return self

    def table(self) -> pd.DataFrame:
        df = pd.DataFrame({"team": list(self.ratings), "elo": list(self.ratings.values())})
        return df.sort_values("elo", ascending=False).reset_index(drop=True)
