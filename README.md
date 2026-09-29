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
3. **Elo second opinion.** A margin-adjusted Elo fitted on nflverse history. It is reported
   next to each candidate, not used to generate bets. See the backtest section for why.

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
python -m sportsbet backtest             # walk-forward Elo vs closing lines, 2015 onward
python -m sportsbet odds --fixture tests/fixtures/odds_api_nfl_sample.json --scan --no-model
```

Every odds pull is appended to `data/sportsbet.duckdb` so line movement is kept. Scan
results land in the `candidates` table. A `bets` table is ready for logging what you actually
place along with the closing price, because closing line value is the only trustworthy
short-term measure of whether this is working.

## Backtest, honestly

Walk-forward Elo on 2,942 regular season games, 2015 to week 3 of 2026, against the
nflverse closing line:

| Metric | Elo | Closing line |
| --- | --- | --- |
| Brier score (lower is better) | 0.223 | 0.212 |
| Log loss | 0.639 | 0.614 |

Betting every game where Elo disagreed with the market by 3% (moneyline) or 2 points
(spread) lost 6.5% and 3.9% per bet respectively. A ratings model built from public box
scores does not beat the closing line. That is the expected result and it is why the
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
  backtest.py        walk-forward evaluation vs closing lines
  store.py           DuckDB tables: odds_snapshots, injuries, candidates, bets
  providers/
    odds_api.py      The Odds API client with quota guard
    nflverse.py      schedules, injuries, depth charts
    espn.py          live injuries feed
  model/
    elo.py           margin-adjusted Elo
    injuries.py      position-weighted injury -> points heuristic
  cli.py
tests/               unit tests plus recorded API fixtures
```

## Known limits and next steps

- Spreads and totals are compared only at the same point. Converting across half points
  needs a push chart built from historical margins (nflverse has the data).
- The injury heuristic weights are hand-set. Fitting them from line moves around
  announced injuries is the obvious upgrade, and QB changes dominate everything else.
- The official report lags. The ESPN feed is the live signal; a polling loop that alerts
  when a starter's status changes and the target book's line has not moved is the next
  feature.
- No player props yet. Props are where soft books are softest, but they cost extra
  Odds API credits per market.
- Books limit winning accounts. Expect BetMGM and Caesars to cut stakes if this works.
