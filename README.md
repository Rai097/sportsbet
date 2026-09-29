# sportsbet

NFL +EV bet finder for **BetMGM** and **Caesars**, built only on free data.

## How it finds edge

1. **Market reference (primary).** Pull prices from The Odds API for BetMGM, Caesars and a
   set of reference books. De-vig Pinnacle's two-way market (power method) to get a fair
   probability; fall back to a de-vigged consensus of DraftKings, FanDuel, BetRivers, Bovada
   and BetOnline when Pinnacle has no line. Any BetMGM or Caesars price that pays more than
   fair is a candidate, ranked by EV with a fractional-Kelly stake.
2. **Injury latency.** ESPN's live injuries feed plus nflverse's official weekly report are
   turned into a point-spread adjustment per team using depth charts to decide who is a
   starter. The goal is to spot a status change before the soft books have repriced.
3. **Model second opinion.** A margin-adjusted Elo with a quarterback layer: each starting
   QB carries an EPA-based value, so a backup starting moves the line. It is reported next
   to each candidate, not used to generate bets. See the backtest section for why.
4. **Closing line value.** Every bet you log gets its closing price and no-vig closing
   probability captured automatically, then settled from results. CLV is the scoreboard.

## Free data sources

| Data | Source | Cost | Notes |
| --- | --- | --- | --- |
| Live odds (MGM, Caesars, Pinnacle, others) | [The Odds API](https://the-odds-api.com) | Free tier, 500 credits/month | Client tracks quota headers and refuses to spend below a floor. 3 markets x up to 10 books = 3 credits per pull, about 5 pulls a day. |
| Schedules, results, closing lines since 2010 | nflverse via `nflreadpy` | Free | Also the free reference line for the current week. |
| Official injury reports, depth charts | nflverse via `nflreadpy` | Free | Reports publish Wed-Fri only. |
| Live injuries | ESPN public JSON | Free, unofficial | Parser is tolerant; endpoint may change. |

## Setup

```bash
pip install -e ".[dev]"
export ODDS_API_KEY=...   # free key from the-odds-api.com
python -m pytest
```

## Commands

```bash
python -m sportsbet week                 # this week's slate: market line vs Elo line
python -m sportsbet odds --scan          # pull live odds (3 credits), store, and print +EV bets
python -m sportsbet scan --min-ev 0.02   # re-scan stored odds without spending credits
python -m sportsbet injuries             # official report -> per-team point impact
python -m sportsbet injuries --source espn
python -m sportsbet backtest             # walk-forward Elo and QB-Elo vs closing lines, 2015 onward
python -m sportsbet odds --fixture tests/fixtures/odds_api_nfl_sample.json --scan --no-model

python -m sportsbet alerts               # injury status changes and stale lines from stored data
python -m sportsbet watch --interval 300 --odds-every 3600 --max-odds-pulls-per-day 4
python -m sportsbet watch --fixture-dir tests/fixtures/watch --interval 0   # offline demo

python -m sportsbet bet add --game "NE @ BUF" --book mgm --market h2h --outcome NE --stake 1
python -m sportsbet bet list --open
python -m sportsbet bet settle           # capture closing lines, grade from results
python -m sportsbet clv                  # ROI and closing line value by book and market

python -m sportsbet report               # markdown report under reports/
python -m sportsbet run-tick             # one unattended tick (what GitHub Actions runs)
```

## Running it for free on GitHub Actions

The workflow in `.github/workflows/sportsbet.yml` runs hourly, pulls odds only in budgeted
windows (about 90 minutes before each kickoff slate plus Wednesday and Friday afternoon
for injury reports, at most 6 pulls a week, roughly 80 credits a month), persists the DuckDB
history between runs, and commits `reports/latest.md`. Setup steps are in
[docs/actions.md](docs/actions.md). The one thing it needs from you is the repository
secret `ODDS_API_KEY`.

Every odds pull is appended to `data/sportsbet.duckdb` so line movement is kept. Scan
results land in the `candidates` table. A `bets` table is ready for logging what you actually
place along with the closing price, because closing line value is the only trustworthy
short-term measure of whether this is working.

## Backtest, honestly

Walk-forward Elo on 2,942 regular season games, 2015 to week 3 of 2026, against the
nflverse closing line:

| Metric | Elo | QB-aware Elo | Closing line |
| --- | --- | --- | --- |
| Brier score (lower is better) | 0.223 | 0.219 | 0.212 |
| Log loss | 0.639 | 0.629 | 0.614 |

On a 2019 onward holdout (QB hyperparameters were tuned on 2011 to 2018 and frozen) the
Brier scores are 0.2243 plain, 0.2207 QB-aware, 0.2106 market. The QB layer is a real,
measurable improvement and still clearly short of the market.

Betting every game where Elo disagreed with the market by 3% (moneyline) or 2 points
(spread) lost 6.5% and 3.9% per bet respectively. The QB-aware model narrows the moneyline
loss to 3.6% and still loses. A ratings model built from public box scores does not beat
the closing line. That is the expected result and it is why the
primary signal here is the price difference between the soft books and the sharp reference,
which needs no forecasting at all, plus reacting to injury news faster than the books.

Historical BetMGM and Caesars prices are not free, so the book-shopping edge cannot be
backtested here. It is measured going forward via the `bets` table and closing line value.

## Layout

```
sportsbet/
  config.py          settings, target and reference book keys
  pricing.py         odds conversion, de-vig, EV, Kelly
  engine.py          +EV scan against fair prices
  alerts.py          injury status changes, stale lines, prioritised alerts
  poller.py          watch loop with odds credit budget
  tracking.py        bet log, closing line capture, settlement, CLV report
  report.py          markdown report and the unattended run-tick
  schedule.py        budget-aware odds pull windows around kickoffs
  backtest.py        walk-forward evaluation vs closing lines
  store.py           DuckDB tables: odds_snapshots, injuries, candidates, bets, bet_meta, alerts, pull_log
  providers/
    odds_api.py      The Odds API client with quota guard
    nflverse.py      schedules, injuries, depth charts
    espn.py          live injuries feed
  model/
    elo.py           margin-adjusted Elo
    qb.py            EPA-based starting QB value and QbEloModel
    injuries.py      position-weighted injury -> points; QB weight measured from closing lines
  cli.py
tests/               unit tests plus recorded API fixtures
```

## Known limits and next steps

- Spreads and totals are compared only at the same point. Converting across half points
  needs a push chart built from historical margins (nflverse has the data).
- The QB injury weight is measured: closing lines moved 3.76 points on average when an
  established starter was replaced by a backup (214 games, 2011 to 2026), so a QB counts
  3.8 points or 0.9 points per unit of value gap to the next QB up when both are rated.
  Every other position weight is still hand-set.
- The official injury report lags. The ESPN feed is the live signal and the watch loop
  alerts when a starter's status changes and the target book has not moved.
- The "close" for CLV is only as fresh as the last odds pull before kickoff, so the
  pull windows sit about 90 minutes before each slate.
- No player props yet. Props are where soft books are softest, but they cost extra
  Odds API credits per market.
- Books limit winning accounts. Expect BetMGM and Caesars to cut stakes if this works.
