"""Push chart: convert a spread or total quoted at one point into a probability at another.

Model (a "tilted normal"): P(X = x) is proportional to phi((x - loc) / sigma) * k(x). Each
2010-2025 game's loc is pinned so the model reproduces its de-vigged closing price, and
k(x) is fitted by iterative proportional fitting to the empirical scores. Why: a pure
normal smears key numbers away and residual shifting keeps them only for lines sitting
on a key; the tilt keeps 3/7/6/10/14 at any line, is monotone in loc, and is calibrated
to the market the same way the scan uses it (a price at one point -> a fair line).

Conventions follow the odds feed: a spread `point` is the side's own handicap (home -7
means the home team must win by 8+, pushes at 7), and `fair_line` is the fair home
handicap at which the home side is a 50% (push-excluded) proposition, interpolated
between half points, so -7.0 means the home team is a fair 7 point favourite.
cover_prob/over_prob exclude pushes (win / (win + loss)), matching how a two-way de-vig
treats an integer line; spread_probs/total_probs give (win, push, loss).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from sportsbet.config import DATA_DIR
from sportsbet.pricing import fair_prob_from_two_way

log = logging.getLogger(__name__)

CACHE_PATH = DATA_DIR / "cache" / "pushchart.json"
SEASONS = range(2010, 2026)
CACHE_VERSION = 2
MARGIN_MAX = 80  # support is [-80, 80]; the largest NFL margin since 2010 is under 60
TOTAL_MAX = 130
# Fraction of the IPF target taken from a plain normal instead of the observed counts.
# Without it a score seen once in the tails gets a weight fitted to that one game.
# Chosen by out-of-sample log loss (fit 2010-2021, score 2022-2025).
SHRINK_MARGIN = 0.1
SHRINK_TOTAL = 0.25
SIGMA_GRID = np.round(np.arange(11.0, 17.01, 0.5), 2)


def _kernel(support: np.ndarray, locs: np.ndarray, sigma: float) -> np.ndarray:
    """Unnormalised normal kernel, one row per location."""
    z = (support[None, :] - locs[:, None]) / sigma
    return np.exp(-0.5 * z * z)


def _pmf_rows(support: np.ndarray, weights: np.ndarray, sigma: float, locs: np.ndarray) -> np.ndarray:
    p = _kernel(support, locs, sigma) * weights
    return p / p.sum(axis=1, keepdims=True)


def _solve_locs(
    support: np.ndarray, weights: np.ndarray, sigma: float, xs: np.ndarray, ps: np.ndarray, iters: int = 28
) -> np.ndarray:
    """Locations at which P(X > x | X != x) equals p, one per (x, p), by vectorised bisection.

    The tilted family has a monotone likelihood ratio in loc, so the push-excluded
    probability of clearing any fixed x rises with loc and bisection is safe.
    """
    ps = np.clip(ps, 1e-6, 1 - 1e-6)
    lo, hi = xs - 20.0, xs + 20.0
    above_mask = support[None, :] > xs[:, None]
    below_mask = support[None, :] < xs[:, None]
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        pmf = _pmf_rows(support, weights, sigma, mid)
        a = (pmf * above_mask).sum(axis=1)
        b = (pmf * below_mask).sum(axis=1)
        short = a / (a + b) < ps
        lo = np.where(short, mid, lo)
        hi = np.where(short, hi, mid)
    return (lo + hi) / 2.0


@dataclass
class Dist:
    """Tilted-normal distribution for one quantity (home margin or game total).

    `loc` is the kernel centre, an internal parameter. Callers work with the fair line:
    the point, interpolated on the half-point grid, at which the push-excluded
    probability of going over it is exactly 50%.
    """

    support: np.ndarray
    weights: np.ndarray
    sigma: float
    counts: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def pmf(self, loc: float) -> np.ndarray:
        return _pmf_rows(self.support, self.weights, self.sigma, np.array([float(loc)]))[0]

    def above_eq_below(self, loc: float, x: float) -> tuple[float, float, float]:
        """(P(X > x), P(X == x), P(X < x))."""
        p = self.pmf(loc)
        s = self.support
        return float(p[s > x].sum()), float(p[s == x].sum()), float(p[s < x].sum())

    def loc_for_prob(self, x: float, p_above: float) -> float:
        """loc at which P(X > x | X != x) equals p_above."""
        return float(_solve_locs(self.support, self.weights, self.sigma, np.array([float(x)]), np.array([p_above]))[0])

    def line_at(self, loc: float) -> float:
        """Fair line (50% point) of the distribution centred at `loc`."""
        cdf = np.cumsum(self.pmf(loc))
        xs = np.floor(loc * 2) / 2 + np.arange(-16, 17) * 0.5
        n_below = np.searchsorted(self.support, xs, side="left")
        n_at_or_below = np.searchsorted(self.support, xs, side="right")
        below = np.where(n_below > 0, cdf[np.maximum(n_below - 1, 0)], 0.0)
        above = 1.0 - np.where(n_at_or_below > 0, cdf[np.maximum(n_at_or_below - 1, 0)], 0.0)
        # strictly decreasing in x as long as every weight is positive
        c = above / (above + below)
        j = int(np.nonzero(c >= 0.5)[0][-1])
        return float(xs[j] + 0.5 * (c[j] - 0.5) / (c[j] - c[j + 1]))

    def loc_for_line(self, line: float) -> float:
        lo, hi = line - 12.0, line + 12.0
        for _ in range(50):
            mid = (lo + hi) / 2.0
            if self.line_at(mid) < line:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    def pmf_at_line(self, line: float) -> np.ndarray:
        return self.pmf(self.loc_for_line(line))

    def probs_at_line(self, line: float, x: float) -> tuple[float, float, float]:
        """(P(X > x), P(X == x), P(X < x)) when the fair line is `line`."""
        return self.above_eq_below(self.loc_for_line(line), x)

    def line_from_prob(self, x: float, p_above: float) -> float:
        """Fair line implied by a push-excluded probability of clearing `x`."""
        return self.line_at(self.loc_for_prob(x, p_above))

    def to_json(self) -> dict:
        return {
            "support_min": int(self.support[0]),
            "sigma": self.sigma,
            "weights": [float(w) for w in self.weights],
            "counts": [int(c) for c in self.counts],
        }

    @classmethod
    def from_json(cls, d: dict) -> Dist:
        w = np.array(d["weights"], dtype=float)
        support = np.arange(d["support_min"], d["support_min"] + len(w), dtype=float)
        return cls(support, w, float(d["sigma"]), np.array(d.get("counts", []), dtype=float))


def _ipf(
    kern: np.ndarray,
    n_per_row: np.ndarray,
    target: np.ndarray,
    symmetric: bool,
    w: np.ndarray | None = None,
    max_iter: int = 2000,
    tol: float = 1e-6,
) -> np.ndarray:
    """Weights w so that summed P(x | row) over games reproduces the `target` counts."""
    w = np.ones(kern.shape[1]) if w is None else w.copy()
    live = target > 0.5
    for _ in range(max_iter):
        p = kern * w
        p /= p.sum(axis=1, keepdims=True)
        expected = (p * n_per_row[:, None]).sum(axis=0)
        if symmetric:
            expected = expected + expected[::-1]
        ratio = (target + 1e-12) / (expected + 1e-12)
        w = w * ratio
        w /= w.max()
        if np.max(np.abs(ratio - 1.0)[live]) < tol:
            break
    return w


def fit_dist(
    lines: np.ndarray,
    p_above: np.ndarray,
    outcomes: np.ndarray,
    support: np.ndarray,
    symmetric: bool,
    shrink: float = SHRINK_MARGIN,
    sigma_grid: np.ndarray = SIGMA_GRID,
    outer: int = 12,
) -> Dist:
    """Fit weights and sigma, conditioning each game on its de-vigged closing price.

    Each game's loc is set so the model reproduces the closing market's push-excluded
    probability of clearing the closing line; the weights are then refit by IPF, and
    the two steps alternate. Sigma is picked by likelihood on a grid.
    symmetric=True ties k(x) to k(-x): home edge lives in loc, not in the key numbers.
    """
    lines = np.asarray(lines, dtype=float)
    p_above = np.nan_to_num(np.asarray(p_above, dtype=float), nan=0.5)
    outcomes = np.asarray(outcomes, dtype=int)
    support = np.asarray(support, dtype=float)
    idx = outcomes - int(support[0])
    counts = np.bincount(idx, minlength=len(support)).astype(float)
    observed = counts + counts[::-1] if symmetric else counts
    # group games by (line, price) so the matrices are (combos x support), not (games x support)
    combos, inv, n_per = np.unique(
        np.c_[lines, np.round(p_above * 200) / 200], axis=0, return_inverse=True, return_counts=True
    )
    inv = inv.ravel()
    xs, ps = combos[:, 0], combos[:, 1]

    best: tuple[float, Dist] | None = None
    for sigma in sigma_grid:
        sigma = float(sigma)
        w = np.ones(len(support))
        locs = _solve_locs(support, w, sigma, xs, ps)
        normal_counts = (_pmf_rows(support, w, sigma, locs) * n_per[:, None]).sum(axis=0)
        if symmetric:
            normal_counts = normal_counts + normal_counts[::-1]
        target = (1.0 - shrink) * observed + shrink * normal_counts
        for _ in range(outer):
            w = _ipf(_kernel(support, locs, sigma), n_per, target, symmetric, w)
            new_locs = _solve_locs(support, w, sigma, xs, ps)
            moved = np.max(np.abs(new_locs - locs))
            locs = new_locs
            if moved < 1e-3:
                break
        p = _pmf_rows(support, w, sigma, locs)
        loglik = float(np.log(np.maximum(p[inv, idx], 1e-300)).sum())
        if best is None or loglik > best[0]:
            best = (loglik, Dist(support, w, sigma, counts))
    assert best is not None
    return best[1]


@dataclass
class PushChart:
    margin: Dist  # home margin (home score - away score)
    total: Dist
    n_games: int = 0
    seasons: tuple[int, int] = (0, 0)

    # ---- spreads -------------------------------------------------------------------
    def margin_pmf(self, fair_line: float) -> pd.Series:
        """P(home margin == m) for a fair home handicap (e.g. -7.0 = home favoured by 7)."""
        return pd.Series(self.margin.pmf_at_line(-fair_line), index=self.margin.support.astype(int))

    def spread_probs(self, fair_line: float, point: float, side: str = "home") -> tuple[float, float, float]:
        """(win, push, loss) for `side` laying/taking its own `point`."""
        if side == "home":
            # home -7 wins when the home margin is above 7
            win, push, loss = self.margin.probs_at_line(-fair_line, -point)
        elif side == "away":
            # away +7 wins when the home margin is below 7
            loss, push, win = self.margin.probs_at_line(-fair_line, point)
        else:
            raise ValueError(f"side must be 'home' or 'away', got {side!r}")
        return win, push, loss

    def cover_prob(self, fair_line: float, point: float, side: str = "home") -> float:
        """P(side covers at its own `point`), pushes excluded."""
        win, _, loss = self.spread_probs(fair_line, point, side)
        return win / (win + loss)

    def fair_spread_from_prob(self, point: float, p_home_cover: float) -> float:
        """Fair home handicap implied by a de-vigged home cover probability at home `point`."""
        return -self.margin.line_from_prob(-point, p_home_cover)

    def fair_spread_from_prices(self, point: float, price_home: float, price_away: float) -> float:
        """Fair home handicap from a two-way spread (home at `point`, away at -`point`)."""
        p_home, _ = fair_prob_from_two_way(price_home, price_away)
        return self.fair_spread_from_prob(point, p_home)

    # ---- totals --------------------------------------------------------------------
    def total_pmf(self, fair_total: float) -> pd.Series:
        return pd.Series(self.total.pmf_at_line(fair_total), index=self.total.support.astype(int))

    def total_probs(self, fair_total: float, point: float) -> tuple[float, float, float]:
        """(over, push, under) at `point`."""
        return self.total.probs_at_line(fair_total, point)

    def over_prob(self, fair_total: float, point: float) -> float:
        """P(over at `point`), pushes excluded. Under is 1 minus this."""
        over, _, under = self.total_probs(fair_total, point)
        return over / (over + under)

    def fair_total_from_prob(self, point: float, p_over: float) -> float:
        return self.total.line_from_prob(point, p_over)

    def fair_total_from_prices(self, point: float, price_over: float, price_under: float) -> float:
        p_over, _ = fair_prob_from_two_way(price_over, price_under)
        return self.fair_total_from_prob(point, p_over)

    # ---- persistence ---------------------------------------------------------------
    def to_json(self) -> dict:
        return {
            "version": CACHE_VERSION,
            "n_games": self.n_games,
            "seasons": list(self.seasons),
            "margin": self.margin.to_json(),
            "total": self.total.to_json(),
        }

    @classmethod
    def from_json(cls, d: dict) -> PushChart:
        return cls(Dist.from_json(d["margin"]), Dist.from_json(d["total"]), int(d["n_games"]), tuple(d["seasons"]))

    def save(self, path: Path = CACHE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json()))


def _devig_columns(df: pd.DataFrame, col_a: str, col_b: str) -> np.ndarray:
    """De-vigged probability of side a per row; 0.5 where a price is missing."""
    out = np.full(len(df), 0.5)
    if col_a not in df.columns or col_b not in df.columns:
        return out
    for i, (pa, pb) in enumerate(zip(df[col_a].to_numpy(), df[col_b].to_numpy())):
        if pd.notna(pa) and pd.notna(pb) and pa != 0 and pb != 0:
            out[i] = fair_prob_from_two_way(pa, pb)[0]
    return out


def fit(schedule: pd.DataFrame) -> PushChart:
    """Fit margin and total distributions on completed regular-season games with closing lines."""
    g = schedule[
        (schedule["game_type"] == "REG")
        & schedule["result"].notna()
        & schedule["spread_line"].notna()
        & schedule["total_line"].notna()
        & schedule["result"].abs().le(MARGIN_MAX)
        & schedule["total"].le(TOTAL_MAX)
    ]
    # spread_line is the expected home margin, so the home side covers when result > spread_line
    margin = fit_dist(
        g["spread_line"].to_numpy(), _devig_columns(g, "home_spread_odds", "away_spread_odds"),
        g["result"].to_numpy(), np.arange(-MARGIN_MAX, MARGIN_MAX + 1), symmetric=True, shrink=SHRINK_MARGIN,
    )
    total = fit_dist(
        g["total_line"].to_numpy(), _devig_columns(g, "over_odds", "under_odds"),
        g["total"].to_numpy(), np.arange(0, TOTAL_MAX + 1), symmetric=False, shrink=SHRINK_TOTAL,
    )
    return PushChart(margin, total, len(g), (int(g["season"].min()), int(g["season"].max())))


def build(path: Path = CACHE_PATH, seasons: range = SEASONS) -> PushChart:
    """Load nflverse schedules, fit, and write the cache."""
    from sportsbet.providers import nflverse

    chart = fit(nflverse.load_schedules(list(seasons)))
    chart.save(path)
    log.info("push chart fitted on %d games, cached to %s", chart.n_games, path)
    return chart


def load(path: Path = CACHE_PATH, rebuild: bool = False) -> PushChart:
    """Read the cached chart, building it from nflverse if missing or from an older version."""
    if not rebuild and path.exists():
        try:
            d = json.loads(path.read_text())
            if d.get("version") == CACHE_VERSION:
                return PushChart.from_json(d)
        except (ValueError, KeyError) as exc:
            log.warning("unreadable push chart cache %s: %s", path, exc)
    return build(path)


_default: PushChart | None = None


def default_chart() -> PushChart:
    """The cached chart, loaded once per process."""
    global _default
    if _default is None:
        _default = load()
    return _default


# Module-level shortcuts on the cached chart.
def fair_spread_from_prices(point: float, price_a: float, price_b: float) -> float:
    """Fair home handicap given the home side at `point` priced price_a, away at price_b."""
    return default_chart().fair_spread_from_prices(point, price_a, price_b)


def cover_prob(fair_line: float, point: float, side: str = "home") -> float:
    return default_chart().cover_prob(fair_line, point, side)


def fair_total_from_prices(point: float, price_over: float, price_under: float) -> float:
    return default_chart().fair_total_from_prices(point, price_over, price_under)


def over_prob(fair_total: float, point: float) -> float:
    return default_chart().over_prob(fair_total, point)


# ---- reporting / CLI ---------------------------------------------------------------
def top_margins(chart: PushChart, n: int = 10) -> pd.DataFrame:
    """Most common absolute final margins in the fitted sample."""
    s = pd.Series(chart.margin.counts, index=chart.margin.support.astype(int))
    by_abs = s.groupby(s.index.map(abs)).sum()
    out = (by_abs / by_abs.sum()).sort_values(ascending=False).head(n)
    return out.rename("freq").rename_axis("margin").reset_index()


def half_point_table(chart: PushChart, keys: tuple[int, ...] = (3, 7)) -> pd.DataFrame:
    """Favourite's win/push/loss around each key number when the fair line sits on it.

    `dwin` is the change in outright win probability from the row above, i.e. the value
    of that half point; `cover` excludes pushes and is what a two-way price reflects.
    """
    rows = []
    for key in keys:
        fair = -float(key)
        prev = None
        for pt in np.arange(-(key + 1.0), -(key - 1.0) + 0.01, 0.5):
            win, push, loss = chart.spread_probs(fair, float(pt), "home")
            rows.append(
                {
                    "fair": f"{fair:+.1f}",
                    "point": f"{pt:+.1f}",
                    "win": round(win, 4),
                    "push": round(push, 4),
                    "loss": round(loss, 4),
                    "cover": round(win / (win + loss), 4),
                    "dwin": None if prev is None else round(win - prev, 4),
                }
            )
            prev = win
    return pd.DataFrame(rows)


def cmd_pushchart(args) -> int:
    """Fit (or load) the push chart and show key-number margins and half-point values."""
    chart = load(rebuild=args.rebuild)
    print(
        f"push chart: {chart.n_games} regular-season games {chart.seasons[0]}-{chart.seasons[1]}, "
        f"sigma margin={chart.margin.sigma:.2f} total={chart.total.sigma:.2f}"
    )
    print("\ntop margins (absolute, empirical):")
    tm = top_margins(chart)
    tm["freq"] = tm["freq"].map(lambda v: f"{v:.1%}")
    print(tm.to_string(index=False))
    print("\nhalf-point value, home favourite, fair line on the key:")
    print(half_point_table(chart).to_string(index=False))
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    s = subparsers.add_parser("pushchart", help=cmd_pushchart.__doc__)
    s.add_argument("--rebuild", action="store_true", help="refit from nflverse even if cached")
    s.set_defaults(func=cmd_pushchart)
