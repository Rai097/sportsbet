"""Command line entry point. Run `python -m sportsbet --help`."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from sportsbet import backtest as bt
from sportsbet import tracking
from sportsbet.config import load_settings
from sportsbet.engine import format_candidates, scan
from sportsbet.model.elo import EloModel
from sportsbet.model.injuries import (
    from_nflverse_report,
    matchup_adjustment,
    starters_from_depth_chart,
    team_impacts,
)
from sportsbet.providers import nflverse
from sportsbet.providers.espn import fetch_espn_injuries
from sportsbet.providers.odds_api import OddsApiClient, load_fixture, normalize
from sportsbet.store import Store

log = logging.getLogger("sportsbet")


def _store(settings) -> Store:
    return Store(settings.db_path)


def cmd_odds(args) -> int:
    """Fetch current NFL odds (or load a fixture) and append a snapshot to the store."""
    settings = load_settings()
    if args.fixture:
        rows = normalize(load_fixture(Path(args.fixture)))
    else:
        client = OddsApiClient(settings)
        rows = client.fetch_and_normalize()
        print(f"quota: used={client.quota.used} remaining={client.quota.remaining} last_cost={client.quota.last_cost}")
    store = _store(settings)
    n = store.insert_rows("odds_snapshots", rows)
    print(f"stored {n} odds rows across {len({r.event_id for r in rows})} games")
    if args.scan:
        return cmd_scan(args)
    return 0


def _model_probs(store: Store, odds: pd.DataFrame, use_injuries: bool) -> dict[tuple[str, str], float]:
    """Elo (+ injury heuristic) win probabilities keyed by (event_id, team)."""
    season = nflverse.current_season()
    sched = nflverse.load_schedules(list(range(season - 6, season + 1)))
    model = EloModel().fit(sched)
    impacts = {}
    if use_injuries:
        week = nflverse.current_week(sched[sched["season"] == season])
        inj = store.latest_injuries()
        if inj.empty:
            report = nflverse.load_injuries([season])
            inj = from_nflverse_report(report, season, week)
            if inj.empty and week > 1:  # this week's report may not be out yet
                inj = from_nflverse_report(report, season, week - 1)
        try:
            starters = starters_from_depth_chart(nflverse.load_depth_charts([season]), season, week)
        except Exception as exc:  # depth charts are best-effort
            log.warning("depth charts unavailable: %s", exc)
            starters = None
        impacts = team_impacts(inj, starters)
    out: dict[tuple[str, str], float] = {}
    for ev in odds[["event_id", "home_team", "away_team"]].drop_duplicates().itertuples(index=False):
        adj = matchup_adjustment(impacts, ev.home_team, ev.away_team) if impacts else 0.0
        p_home = model.predict(ev.home_team, ev.away_team, extra_points=adj)["home_win_prob"]
        out[(ev.event_id, ev.home_team)] = p_home
        out[(ev.event_id, ev.away_team)] = 1.0 - p_home
    return out


def cmd_scan(args) -> int:
    """Scan the latest stored odds for +EV prices at BetMGM and Caesars."""
    settings = load_settings()
    store = _store(settings)
    odds = store.latest_odds()
    if odds.empty:
        print("No odds stored yet. Run `sportsbet odds` first.")
        return 1
    model_probs = None
    if not getattr(args, "no_model", False):
        try:
            model_probs = _model_probs(store, odds, use_injuries=not getattr(args, "no_injuries", False))
        except Exception as exc:
            log.warning("model unavailable, market-only scan: %s", exc)
    cands = scan(odds, min_ev=args.min_ev if args.min_ev is not None else settings.min_ev,
                 kelly_frac=settings.kelly_fraction, model_probs=model_probs)
    print(format_candidates(cands))
    store.insert_rows("candidates", cands)
    return 0


def cmd_injuries(args) -> int:
    """Pull injuries (ESPN live feed or the official weekly report) and show team impacts."""
    settings = load_settings()
    store = _store(settings)
    season = nflverse.current_season()
    if args.source == "espn":
        rows = fetch_espn_injuries(settings.espn_injuries_url)
        store.insert_rows("injuries", rows)
        inj = pd.DataFrame([r.as_dict() for r in rows])
        week = None
    else:
        sched = nflverse.load_schedules([season])
        week = args.week or nflverse.current_week(sched)
        inj = from_nflverse_report(nflverse.load_injuries([season]), season, week)
        if inj.empty and week > 1:
            print(f"No official report for week {week} yet; showing week {week - 1}.")
            week -= 1
            inj = from_nflverse_report(nflverse.load_injuries([season]), season, week)
    try:
        starters = starters_from_depth_chart(nflverse.load_depth_charts([season]), season, week)
    except Exception as exc:
        log.warning("depth charts unavailable: %s", exc)
        starters = None
    impacts = team_impacts(inj, starters)
    for team, imp in sorted(impacts.items(), key=lambda kv: -kv[1].points):
        print(f"{team:<4} -{imp.points:.1f} pts  " + "; ".join(imp.detail[:6]))
    return 0


def cmd_week(args) -> int:
    """Show this week's slate with nflverse reference lines and the model's view."""
    season = nflverse.current_season()
    sched = nflverse.load_schedules(list(range(season - 6, season + 1)))
    cur = sched[sched["season"] == season]
    week = args.week or nflverse.current_week(cur)
    model = EloModel().fit(sched)
    games = nflverse.week_games(sched, season, week)
    rows = []
    for g in games.itertuples(index=False):
        pred = model.predict(g.home_team, g.away_team)
        rows.append(
            {
                "date": g.gameday.date(),
                "game": f"{g.away_team} @ {g.home_team}",
                "mkt_spread": g.spread_line,
                "elo_spread": round(pred["home_spread"], 1),
                "diff": round(pred["home_spread"] - g.spread_line, 1) if pd.notna(g.spread_line) else None,
                "home_ml": g.home_moneyline,
                "elo_home_p": f"{pred['home_win_prob']:.1%}",
            }
        )
    print(f"{season} week {week}  (mkt_spread is home margin, positive = home favoured)")
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


def cmd_backtest(args) -> int:
    """Walk-forward Elo vs closing lines on nflverse history."""
    sched = nflverse.load_schedules(list(range(args.start, args.end + 1)))
    res = bt.evaluate(sched, eval_from=args.eval_from, ml_edge=args.ml_edge, ats_edge=args.ats_edge)
    print(res.summary())
    print("\ncalibration (model):")
    print(bt.calibration_table(res.games).to_string())
    if res.bets is not None and not res.bets.empty:
        print("\nROI by season:")
        print(res.bets.groupby(["type", "season"])["pnl"].agg(["count", "mean"]).round(3).to_string())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sportsbet", description="NFL +EV finder for BetMGM and Caesars (free data only)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("odds", help=cmd_odds.__doc__)
    s.add_argument("--fixture", help="load a saved Odds API JSON instead of calling the API")
    s.add_argument("--scan", action="store_true", help="run the +EV scan after storing")
    s.add_argument("--min-ev", type=float, default=None)
    s.add_argument("--no-model", action="store_true")
    s.add_argument("--no-injuries", action="store_true")
    s.set_defaults(func=cmd_odds)

    s = sub.add_parser("scan", help=cmd_scan.__doc__)
    s.add_argument("--min-ev", type=float, default=None, help="minimum EV per unit (default from settings)")
    s.add_argument("--no-model", action="store_true", help="skip the Elo second opinion")
    s.add_argument("--no-injuries", action="store_true", help="skip the injury adjustment")
    s.set_defaults(func=cmd_scan)

    s = sub.add_parser("injuries", help=cmd_injuries.__doc__)
    s.add_argument("--source", choices=["espn", "official"], default="official")
    s.add_argument("--week", type=int)
    s.set_defaults(func=cmd_injuries)

    s = sub.add_parser("week", help=cmd_week.__doc__)
    s.add_argument("--week", type=int)
    s.set_defaults(func=cmd_week)

    s = sub.add_parser("backtest", help=cmd_backtest.__doc__)
    s.add_argument("--start", type=int, default=2010)
    s.add_argument("--end", type=int, default=date.today().year - (0 if date.today().month >= 3 else 1))
    s.add_argument("--eval-from", type=int, default=2015)
    s.add_argument("--ml-edge", type=float, default=0.03, help="min model-vs-market prob gap to bet a moneyline")
    s.add_argument("--ats-edge", type=float, default=2.0, help="min points of disagreement to bet a spread")
    s.set_defaults(func=cmd_backtest)
    tracking.register(sub)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
