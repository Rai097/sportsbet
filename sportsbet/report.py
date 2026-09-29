"""Markdown report and the unattended `run-tick` job behind the GitHub Actions workflow.

`sportsbet report` renders what a person wants on their phone: quota status, the week's
slate (market line vs model line), +EV candidates from the latest stored odds, alerts
from the last day, injury impact per team, and how fresh the data is. Every section degrades to a short
"not available" note instead of failing, because the report is the only output of an
unattended run and an empty store (no API key yet) is a normal state.

`sportsbet run-tick` is one scheduled tick: maybe spend credits on odds (schedule.py
decides), always try the free ESPN injuries feed, build alerts, then scan and write the report.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from sportsbet import alerts as alerts_mod
from sportsbet.config import TARGET_BOOKS, Settings, load_settings
from sportsbet.engine import Candidate, format_candidates, scan
from sportsbet.model.elo import EloModel
from sportsbet.model.injuries import (
    InjuryImpact,
    from_nflverse_report,
    matchup_adjustment,
    starters_from_depth_chart,
    team_impacts,
)
from sportsbet.model.qb import QbEloModel, load_qb_game_values, qb_injury_gaps
from sportsbet.providers import nflverse
from sportsbet.providers.espn import fetch_espn_injuries
from sportsbet.providers.odds_api import OddsApiClient
from sportsbet.schedule import ET, MAX_PULLS_PER_WEEK, due_slot, next_slot, pulls_this_week, week_bounds
from sportsbet.store import Store

NEAR_MISSES = 5

log = logging.getLogger(__name__)

HISTORY_SEASONS = 6  # seasons of results the Elo fit consumes, same as `sportsbet week`
ESPN_FRESH_FOR = timedelta(hours=48)  # older than this, fall back to the official report
GENERATED_PREFIX = "_Generated "  # the one line allowed to differ between identical reports
CREDITS_PER_PULL = 3
ALERTS_FOR = timedelta(hours=24)  # alerts older than this are news the books have absorbed


@dataclass
class ReportData:
    generated_at: datetime
    season: int
    week: int | None
    min_ev: float
    quota: list[str] = field(default_factory=list)
    slate: pd.DataFrame | None = None
    slate_note: str | None = None
    candidates: list[Candidate] = field(default_factory=list)
    candidates_note: str | None = None
    model_note: str | None = None
    impacts: dict[str, InjuryImpact] | None = None
    injuries_source: str | None = None
    injuries_note: str | None = None
    odds_fetched_at: datetime | None = None
    injuries_fetched_at: datetime | None = None
    alerts: list[alerts_mod.Alert] = field(default_factory=list)
    coverage: pd.DataFrame | None = None
    near_misses: list[Candidate] = field(default_factory=list)


def utcnow() -> datetime:
    """Current time; a function so tests can move the clock into a pull window."""
    return datetime.now(timezone.utc)


def fmt_time(ts: datetime | None) -> str:
    if ts is None:
        return "never"
    ts = ts.astimezone(timezone.utc)
    return f"{ts.astimezone(ET):%a %b %d %H:%M} ET ({ts:%H:%M} UTC)"


def load_history(season: int) -> pd.DataFrame:
    return nflverse.load_schedules(list(range(season - HISTORY_SEASONS, season + 1)))


def slate_frame(
    sched: pd.DataFrame,
    season: int,
    week: int,
    model: EloModel | None,
    impacts: dict[str, InjuryImpact] | None,
) -> pd.DataFrame:
    """This week's games with the nflverse market line and the model line (home margin, + = home favoured).

    Same numbers as `sportsbet week`, plus the injury adjustment folded into the model line.
    """
    games = nflverse.week_games(sched, season, week)
    rows = []
    for g in games.itertuples(index=False):
        row = {
            "kickoff (ET)": f"{pd.Timestamp(g.gameday):%a %m/%d} {g.gametime if pd.notna(g.gametime) else 'TBD'}",
            "game": f"{g.away_team} @ {g.home_team}",
            "market": _num(g.spread_line),
        }
        if model is not None:
            adj = matchup_adjustment(impacts, g.home_team, g.away_team) if impacts else 0.0
            pred = model.predict(g.home_team, g.away_team, extra_points=adj)
            row["inj"] = _num(adj)
            row["model"] = _num(pred["home_spread"])
            row["diff"] = _num(pred["home_spread"] - g.spread_line) if pd.notna(g.spread_line) else ""
            row["home win"] = f"{pred['home_win_prob']:.0%}"
        if pd.notna(g.result):
            row["final"] = _num(g.result)
        rows.append(row)
    return pd.DataFrame(rows).fillna("")


def _num(x) -> str:
    # adding 0.0 turns a rounded -0.0 into 0.0 so it prints as +0.0
    return "" if x is None or pd.isna(x) else f"{round(float(x), 1) + 0.0:+.1f}"


def load_qb_gaps(season: int, sched: pd.DataFrame, depth: pd.DataFrame) -> dict[tuple[str, str], float]:
    """(team, QB1) -> value over the next QB up, so a QB injury is priced by who replaces him."""
    seasons = list(range(season - HISTORY_SEASONS, season + 1))
    qb_model = QbEloModel.from_game_values(load_qb_game_values(seasons, sched)).fit(sched)
    return qb_injury_gaps(qb_model, depth)


def injury_impacts(
    store: Store, season: int, week: int | None, now: datetime, sched: pd.DataFrame | None = None
) -> tuple[dict[str, InjuryImpact], str]:
    """Per-team point impact from the freshest injury source available.

    The live ESPN snapshot wins when recent; otherwise the official nflverse report for
    this week (or last week, before this week's report is published). A QB is priced by
    his value gap to the backup when the QB model can rate both, else the flat weight.
    This is the one place injury points are computed, so `scan` and the report agree.
    """
    espn_at = store.latest_fetch_time("injuries", source="espn")
    if espn_at is not None and now - espn_at <= ESPN_FRESH_FOR:
        inj = store.latest_injury_snapshot("espn")
        source = f"ESPN live feed, fetched {fmt_time(espn_at)}"
    else:
        if week is None:
            raise RuntimeError("no recent ESPN snapshot and the current week is unknown")
        report = nflverse.load_injuries([season])
        inj = from_nflverse_report(report, season, week)
        source = f"official report, week {week}"
        if inj.empty and week > 1:
            inj = from_nflverse_report(report, season, week - 1)
            source = f"official report, week {week - 1} (week {week} not out yet)"
    starters, qb_gaps = None, None
    try:
        depth = nflverse.load_depth_charts([season])
        starters = starters_from_depth_chart(depth, season, week)
    except Exception as exc:  # without depth charts everyone counts as a starter
        log.warning("depth charts unavailable: %s", exc)
        source += "; depth charts unavailable, all listed players treated as starters"
    else:
        try:
            qb_gaps = load_qb_gaps(season, sched if sched is not None else load_history(season), depth)
        except Exception as exc:  # player stats are best-effort; the flat QB weight still applies
            log.warning("QB values unavailable, flat QB weight: %s", exc)
            source += "; QB values unavailable, flat QB weight"
    return team_impacts(inj, starters, qb_gaps=qb_gaps), source


def model_probs(
    model: EloModel, odds: pd.DataFrame, impacts: dict[str, InjuryImpact] | None
) -> dict[tuple[str, str], float]:
    """Elo (+ injury adjustment) win probability keyed by (event_id, team), as the scan expects."""
    out: dict[tuple[str, str], float] = {}
    for ev in odds[["event_id", "home_team", "away_team"]].drop_duplicates().itertuples(index=False):
        adj = matchup_adjustment(impacts, ev.home_team, ev.away_team) if impacts else 0.0
        p_home = model.predict(ev.home_team, ev.away_team, extra_points=adj)["home_win_prob"]
        out[(ev.event_id, ev.home_team)] = p_home
        out[(ev.event_id, ev.away_team)] = 1.0 - p_home
    return out


def quota_lines(store: Store, settings: Settings, now: datetime, sched: pd.DataFrame | None) -> list[str]:
    lines = []
    if settings.odds_api_key:
        lines.append("ODDS_API_KEY: set")
    else:
        lines.append("ODDS_API_KEY: **not set**, odds pulls are skipped (free sources still run)")
    log_df = store.pull_log()
    if log_df.empty:
        lines.append("Paid odds pulls: none logged yet")
    else:
        last = log_df.iloc[-1]
        remaining = "unknown" if pd.isna(last["credits_remaining"]) else int(last["credits_remaining"])
        used = "unknown" if pd.isna(last["credits_used"]) else int(last["credits_used"])
        lines.append(
            f"Last paid pull: {fmt_time(last['pulled_at'].to_pydatetime())} ({last['slot']}); "
            f"API credits remaining {remaining}, used {used} this month"
        )
        month = log_df[log_df["pulled_at"] >= pd.Timestamp(now.year, now.month, 1, tz="UTC")]
        spent = month["credits_last"].fillna(CREDITS_PER_PULL).sum()
        lines.append(f"Pulls logged this month: {len(month)} (~{int(spent)} credits)")
    n_week = pulls_this_week(now, store.pull_times(since=week_bounds(now)[0]))
    lines.append(f"Pulls this NFL week: {n_week} of {MAX_PULLS_PER_WEEK}")
    if sched is not None:
        nxt = next_slot(now, sched)
        lines.append(f"Next planned pull: {nxt.label}, {fmt_time(nxt.at)}" if nxt else "Next planned pull: none scheduled")
    return lines


def gather(
    store: Store,
    settings: Settings,
    now: datetime | None = None,
    sched: pd.DataFrame | None = None,
    min_ev: float | None = None,
) -> ReportData:
    """Collect every report section. Network-dependent pieces fail soft into notes."""
    now = now or utcnow()
    season = nflverse.current_season(now.astimezone(ET).date())
    data = ReportData(
        generated_at=now,
        season=season,
        week=None,
        min_ev=settings.min_ev if min_ev is None else min_ev,
    )

    if sched is None:
        try:
            sched = load_history(season)
        except Exception as exc:
            log.warning("schedules unavailable: %s", exc)
            data.slate_note = f"Not available: could not load the nflverse schedule ({exc})."
    cur = sched[sched["season"] == season] if sched is not None else None
    if cur is not None and not cur.empty:
        data.week = nflverse.current_week(cur, now.astimezone(ET).date())
    elif sched is not None:
        data.slate_note = f"Not available: no {season} games in the nflverse schedule yet."

    model = None
    if sched is not None:
        try:
            model = EloModel().fit(sched)
        except Exception as exc:
            log.warning("model unavailable: %s", exc)
            data.model_note = f"Model unavailable ({exc}); showing market-only numbers."

    try:
        data.impacts, data.injuries_source = injury_impacts(store, season, data.week, now, sched)
    except Exception as exc:
        log.warning("injuries unavailable: %s", exc)
        data.injuries_note = f"Not available: {exc}"

    if cur is not None and not cur.empty and data.week is not None:
        try:
            data.slate = slate_frame(sched, season, data.week, model, data.impacts)
        except Exception as exc:
            log.warning("slate unavailable: %s", exc)
            data.slate_note = f"Not available: {exc}"

    data.quota = quota_lines(store, settings, now, cur)
    data.alerts = recent_alerts(store, now)
    data.odds_fetched_at = store.latest_fetch_time("odds_snapshots")
    data.injuries_fetched_at = store.latest_fetch_time("injuries")

    odds = store.latest_odds()
    if odds.empty:
        data.candidates_note = (
            "Not available: no odds for upcoming games are stored. "
            + ("They arrive at the next planned pull." if settings.odds_api_key else "Add the ODDS_API_KEY secret to start pulling.")
        )
        return data
    probs = None
    if model is not None:
        try:
            probs = model_probs(model, odds, data.impacts)
        except Exception as exc:
            log.warning("model probabilities unavailable: %s", exc)
            data.model_note = f"Model unavailable ({exc}); market-only scan."
    elif data.model_note is None:
        data.model_note = "Model unavailable (no schedule history); market-only scan."
    data.coverage = board_coverage(odds)
    data.candidates = scan(odds, min_ev=data.min_ev, kelly_frac=settings.kelly_fraction, model_probs=probs)
    if not data.candidates:
        # show the scan is alive on a quiet board: the best prices even though none clear the bar
        data.near_misses = scan(odds, min_ev=-1.0, kelly_frac=settings.kelly_fraction, model_probs=probs)[:NEAR_MISSES]
    return data


def board_coverage(odds: pd.DataFrame) -> pd.DataFrame:
    """Games and markets quoted per book in the latest pull, so a thin board is visible."""
    if odds.empty:
        return pd.DataFrame(columns=["book", "games", "markets"])
    g = odds.groupby("bookmaker").agg(games=("event_id", "nunique"), markets=("market", lambda m: ", ".join(sorted(set(m)))))
    g = g.reset_index().rename(columns={"bookmaker": "book"})
    g["book"] = g["book"].map(lambda b: TARGET_BOOKS.get(b, b))
    return g.sort_values(["games", "book"], ascending=[False, True]).reset_index(drop=True)


def recent_alerts(store: Store, now: datetime) -> list[alerts_mod.Alert]:
    """Alerts emitted in the last day for games not yet started, high priority first."""
    df = store.query(
        "SELECT * FROM alerts WHERE created_at >= ? AND commence_time > ? "
        "ORDER BY priority <> 'high', commence_time, matchup, bookmaker, market",
        [now - ALERTS_FOR, now],
    )
    rows = df.astype(object).where(df.notna(), None).to_dict("records")
    return [alerts_mod.Alert(**r) for r in rows]


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def _top_details(impact: InjuryImpact, n: int) -> list[str]:
    """The n most costly entries; team_impacts formats each detail ending in '-<points>'."""
    def pts(detail: str) -> float:
        try:
            return float(detail.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            return 0.0

    return sorted(impact.detail, key=pts, reverse=True)[:n]


def render(data: ReportData) -> str:
    wk = f"week {data.week}" if data.week is not None else "week unknown"
    out = [f"# NFL +EV report: {data.season} {wk}", "", f"{GENERATED_PREFIX}{fmt_time(data.generated_at)}_", ""]

    out += ["## Odds API quota", ""]
    out += [f"- {line}" for line in data.quota]
    out.append("")

    out += ["## +EV candidates (BetMGM, Caesars)", ""]
    if data.candidates_note:
        out.append(data.candidates_note)
    else:
        out.append(f"Minimum EV {data.min_ev:.1%}. Fair price from Pinnacle, else a de-vigged consensus.")
        out += ["", "```", format_candidates(data.candidates), "```"]
        if data.near_misses:
            out += ["", f"Closest to +EV (top {len(data.near_misses)}, below the bar):", "", "```", format_candidates(data.near_misses), "```"]
        if data.coverage is not None and not data.coverage.empty:
            out += ["", "Board coverage in the latest pull:", "", "| book | games | markets |", "|---|---|---|"]
            out += [f"| {r.book} | {r.games} | {r.markets} |" for r in data.coverage.itertuples(index=False)]
            present = set(data.coverage["book"])
            missing = [name for key, name in TARGET_BOOKS.items() if name not in present]
            if missing:
                out += ["", f"Target books with no lines in the feed: {', '.join(sorted(set(missing)))}. "
                        "The odds API is not carrying them for NFL right now, so nothing can be scanned there."]
    if data.model_note:
        out += ["", data.model_note]
    out.append("")

    out += ["## Alerts (last 24h)", ""]
    if data.alerts:
        out.append("Injury news or a reference move the target book has not matched yet. Check the price first.")
        out += ["", "```", alerts_mod.format_alerts(data.alerts), "```"]
    else:
        out.append("No alerts in the last 24 hours.")
    out.append("")

    out += ["## This week's slate", ""]
    if data.slate is not None and not data.slate.empty:
        out.append(
            "Spreads are the home team's expected margin (+ = home favoured). `market` is the "
            "nflverse consensus line, `model` is Elo plus the injury adjustment `inj`."
        )
        out += ["", _md_table(data.slate)]
    else:
        out.append(data.slate_note or "Not available: no games found for this week.")
    out.append("")

    out += ["## Injury impact", ""]
    if data.impacts:
        out += [f"Source: {data.injuries_source}.", ""]
        rows = [
            {"team": t, "pts": f"-{imp.points:.1f}", "biggest losses": "; ".join(_top_details(imp, 3))}
            for t, imp in sorted(data.impacts.items(), key=lambda kv: -kv[1].points)
        ]
        out.append(_md_table(pd.DataFrame(rows)))
    elif data.impacts is not None:
        out.append(f"No meaningful injuries listed ({data.injuries_source}).")
    else:
        out.append(data.injuries_note or "Not available.")
    out.append("")

    out += [
        "---",
        f"Latest odds snapshot: {fmt_time(data.odds_fetched_at)}. "
        f"Latest injury fetch: {fmt_time(data.injuries_fetched_at)}. "
        "Prices move; check the book before betting.",
        "",
    ]
    return "\n".join(out)


def _stable(md: str) -> str:
    return "\n".join(line for line in md.splitlines() if not line.startswith(GENERATED_PREFIX))


def write_report(md: str, data: ReportData, out_dir: Path | str, only_if_changed: bool = False) -> Path | None:
    """Write reports/<season>-wk<week>/<UTC timestamp>.md and refresh reports/latest.md.

    With only_if_changed, a report identical to latest.md apart from its generation time is
    skipped, so an hourly tick with nothing new does not create a commit.
    """
    out_dir = Path(out_dir)
    latest = out_dir / "latest.md"
    if only_if_changed and latest.exists() and _stable(latest.read_text()) == _stable(md):
        return None
    wk = f"{data.week:02d}" if data.week is not None else "NA"
    path = out_dir / f"{data.season}-wk{wk}" / f"{data.generated_at.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md)
    latest.write_text(md)
    return path


def cmd_report(args) -> int:
    """Write a markdown report (quota, slate, +EV candidates, injuries) to reports/."""
    settings = load_settings()
    store = Store(settings.db_path)
    data = gather(store, settings, min_ev=args.min_ev)
    path = write_report(render(data), data, args.out)
    store.close()
    print(f"wrote {path} and {Path(args.out) / 'latest.md'}")
    return 0


def cmd_run_tick(args) -> int:
    """One unattended tick: scheduled odds pull, free injuries pull, scan and report."""
    settings = load_settings()
    store = Store(settings.db_path)
    now = utcnow()
    season = nflverse.current_season(now.astimezone(ET).date())

    sched = None
    try:
        sched = load_history(season)
    except Exception as exc:
        print(f"schedule unavailable, odds pull only if forced: {exc}")

    pulled = _maybe_pull_odds(store, settings, now, sched, season, force=args.force_pull)

    try:
        rows = fetch_espn_injuries(settings.espn_injuries_url)
        store.insert_rows("injuries", rows)
        print(f"ESPN injuries: stored {len(rows)} rows")
    except Exception as exc:  # free and unofficial: a failure must not stop the tick
        print(f"ESPN injuries skipped: {exc}")

    try:
        new = alerts_mod.build_alerts(
            store, starters=alerts_mod.load_starters(), qb_gaps=alerts_mod.load_qb_gaps(), now=now
        )
        print(f"alerts: {len(new)} new")
    except Exception as exc:  # alerts are extra; the report must still be written
        print(f"alerts skipped: {exc}")

    data = gather(store, settings, now=now, sched=sched, min_ev=args.min_ev)
    if pulled:
        # only fresh prices produce new candidates worth keeping for CLV review
        store.insert_rows("candidates", data.candidates)
    print(f"scan: {len(data.candidates)} candidates" if not data.candidates_note else f"scan skipped: {data.candidates_note}")
    path = write_report(render(data), data, args.out, only_if_changed=not (pulled or args.force_pull))
    print(f"report: wrote {path}" if path else "report: unchanged, nothing written")
    store.close()
    return 0


def _maybe_pull_odds(
    store: Store, settings: Settings, now: datetime, sched: pd.DataFrame | None, season: int, force: bool
) -> bool:
    if force:
        slot_label = "forced"
    elif sched is None:
        print("odds pull skipped: no schedule to plan against")
        return False
    else:
        pulls = store.pull_times(since=week_bounds(now)[0])
        slot = due_slot(now, sched[sched["season"] == season], pulls)
        if slot is None:
            if pulls_this_week(now, pulls) >= MAX_PULLS_PER_WEEK:
                print(f"odds pull skipped: weekly cap of {MAX_PULLS_PER_WEEK} pulls reached")
            else:
                print("odds pull skipped: no pull window open")
            return False
        slot_label = slot.label
    if not settings.odds_api_key:
        print(f"odds pull skipped ({slot_label}): ODDS_API_KEY is not set")
        return False
    client = OddsApiClient(settings)
    try:
        rows = client.fetch_and_normalize()
    except Exception as exc:
        print(f"odds pull failed ({slot_label}): {exc}")
        return False
    n = store.insert_rows("odds_snapshots", rows)
    q = client.quota
    store.log_pull(now, slot_label, force, n, q.last_cost, q.used, q.remaining)
    print(f"odds pull ({slot_label}): stored {n} rows; credits remaining={q.remaining} last_cost={q.last_cost}")
    return True


def register(subparsers) -> None:
    s = subparsers.add_parser("report", help=cmd_report.__doc__)
    s.add_argument("--out", default="reports", help="output directory (default: reports)")
    s.add_argument("--min-ev", type=float, default=None, help="minimum EV per unit (default from settings)")
    s.set_defaults(func=cmd_report)


def register_run_tick(subparsers) -> None:
    s = subparsers.add_parser("run-tick", help=cmd_run_tick.__doc__)
    s.add_argument("--force-pull", action="store_true", help="pull odds now regardless of schedule and weekly cap")
    s.add_argument("--out", default="reports", help="report directory (default: reports)")
    s.add_argument("--min-ev", type=float, default=None)
    s.set_defaults(func=cmd_run_tick)
