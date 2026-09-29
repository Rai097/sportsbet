## Running it for free on GitHub Actions

No server needed. `.github/workflows/sportsbet.yml` runs a tick every hour from Wednesday
through Monday (UTC). Each tick decides on its own whether to spend Odds API credits, pulls
the free ESPN injuries feed, re-scans for +EV prices and commits a fresh report to the repo.

### One-time setup

1. **Push this repository to GitHub** and make sure the workflow file is on the default
   branch (`main`). GitHub only runs scheduled workflows from the default branch.
2. **Add your Odds API key as a secret.** Repository → Settings → Secrets and variables →
   Actions → New repository secret. Name: `ODDS_API_KEY`, value: your free key from
   <https://the-odds-api.com>. Without it everything still runs on free data, and the
   report says odds are skipped.
3. **Enable Actions.** Repository → Actions tab. If GitHub shows "Workflows aren't being run
   on this repository", click *I understand my workflows, go ahead and enable them*. On a
   fork, scheduled workflows also have to be enabled once on that tab.
4. **Check write access.** The workflow asks for `contents: write` so it can commit reports.
   If your organization forces read-only tokens, allow read and write under Settings →
   Actions → General → Workflow permissions.
5. **Kick it off once by hand.** Actions → sportsbet → Run workflow. Tick `force_pull` to
   spend one pull (3 credits) right away so the first report has prices in it.

### Where the reports are

- `reports/latest.md` in the repository is always the newest report. Open it in the GitHub
  app or a phone browser; GitHub renders the tables. Bookmark
  `https://github.com/<you>/<repo>/blob/main/reports/latest.md`.
- `reports/<season>-wk<week>/<UTC time>.md` keeps every report that differed from the one
  before it, so you can see how prices and injuries moved during the week.
- Each run also attaches `latest.md` as the `sportsbet-report` artifact (kept 7 days) on the
  run's page under Actions.

A report has: quota status and the next planned pull, +EV candidates at BetMGM and
Caesars, the week's slate with market line vs model line, per-team injury impact, and when
the odds and injuries were last fetched. Prices move, so check the book before betting.

### What it spends

Odds pulls happen about 90 minutes before the first kickoff of each game day (Thursday,
Saturday when there are Saturday games, Sunday early, Sunday late, Monday) and at noon
Eastern on Wednesday and Friday for injury news. There are never more than 6 scheduled pulls
in a Tuesday-to-Monday NFL week; the `pull_log` table in DuckDB is the counter.

| | Per week | Per month |
| --- | --- | --- |
| Odds pulls (3 credits each) | at most 6 | about 26 |
| Odds API credits (500 free) | at most 18 | about 80 |

A manual run with `force_pull` ignores the schedule and the cap, is logged, and counts toward
the cap for the rest of that week. The client also refuses to pull once fewer than 25
credits remain (`SPORTSBET_QUOTA_FLOOR`).

Actions minutes: public repositories are free without limit. A private repository gets 2,000
free minutes a month; about 620 hourly ticks at 1 to 2 billed minutes each fits, but check
Settings → Billing if you add other workflows.

### How history survives between runs

Runners start empty, so the DuckDB file (`data/sportsbet.duckdb`, all odds snapshots,
injuries, candidates and pull log) is restored from `actions/cache` at the start of each
tick and saved at the end. Line movement and closing line value keep working across runs.

GitHub evicts cache entries unused for 7 days, so once a day (the 12:00 UTC tick) and on
every manual run the database is also uploaded as the `sportsbet-duckdb` artifact (kept 30
days). If a tick finds no cache it restores the newest backup artifact automatically. To
inspect the database yourself, download that artifact and open it with `duckdb`.

### Troubleshooting

- **No new reports for a while.** Nothing changed: a tick only commits when the report
  differs (new prices, injury changes, a game kicking off). Check the Actions tab for runs.
- **Scheduled runs stopped.** GitHub pauses schedules in public repositories after 60 days
  without activity. The report commits normally count as activity; re-enable the workflow on
  the Actions tab if it was paused over the offseason.
- **Odds missing from the report.** Look at the run log for `odds pull skipped` or
  `odds pull failed`. The usual causes are a missing `ODDS_API_KEY` secret or the monthly
  quota running low.
- **Runs start late.** GitHub delays scheduled runs when busy. Each pull window is 90
  minutes wide, so an hourly tick usually lands in it; a skipped window is not retried.
