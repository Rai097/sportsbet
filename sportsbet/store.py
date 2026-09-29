"""DuckDB storage for odds snapshots, injuries, scan candidates and tracked bets."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS odds_snapshots (
    fetched_at TIMESTAMPTZ,
    event_id VARCHAR,
    commence_time TIMESTAMPTZ,
    home_team VARCHAR,
    away_team VARCHAR,
    bookmaker VARCHAR,
    market VARCHAR,
    outcome VARCHAR,
    point DOUBLE,
    price DOUBLE,
    last_update TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS injuries (
    fetched_at TIMESTAMPTZ,
    source VARCHAR,
    team VARCHAR,
    player VARCHAR,
    position VARCHAR,
    status VARCHAR,
    detail VARCHAR
);
CREATE TABLE IF NOT EXISTS candidates (
    scanned_at TIMESTAMPTZ,
    event_id VARCHAR,
    commence_time TIMESTAMPTZ,
    matchup VARCHAR,
    bookmaker VARCHAR,
    market VARCHAR,
    outcome VARCHAR,
    point DOUBLE,
    price DOUBLE,
    fair_prob DOUBLE,
    fair_source VARCHAR,
    ev DOUBLE,
    kelly DOUBLE,
    model_prob DOUBLE
);
CREATE TABLE IF NOT EXISTS bets (
    bet_id VARCHAR,
    placed_at TIMESTAMPTZ,
    event_id VARCHAR,
    matchup VARCHAR,
    bookmaker VARCHAR,
    market VARCHAR,
    outcome VARCHAR,
    point DOUBLE,
    price DOUBLE,
    stake DOUBLE,
    fair_prob DOUBLE,
    closing_price DOUBLE,
    closing_fair_prob DOUBLE,
    result VARCHAR,
    pnl DOUBLE
);
-- Side table for bet tracking fields the original bets schema lacks, keyed by bet_id,
-- so the bets columns stay exactly as other code expects them.
CREATE TABLE IF NOT EXISTS bet_meta (
    bet_id VARCHAR,
    commence_time TIMESTAMPTZ,
    source VARCHAR,
    fair_source VARCHAR,
    closing_point DOUBLE,
    closing_fetched_at TIMESTAMPTZ,
    closing_fair_source VARCHAR,
    note VARCHAR
);
CREATE TABLE IF NOT EXISTS alerts (
    created_at TIMESTAMPTZ,
    alert_key VARCHAR,
    priority VARCHAR,
    trigger VARCHAR,
    event_id VARCHAR,
    commence_time TIMESTAMPTZ,
    matchup VARCHAR,
    bookmaker VARCHAR,
    market VARCHAR,
    side VARCHAR,
    point DOUBLE,
    price DOUBLE,
    detail VARCHAR
);
CREATE TABLE IF NOT EXISTS pull_log (
    pulled_at TIMESTAMPTZ,
    slot VARCHAR,
    forced BOOLEAN,
    n_rows INTEGER,
    credits_last INTEGER,
    credits_used INTEGER,
    credits_remaining INTEGER
);
"""


class Store:
    def __init__(self, path: Path | str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(path))
        self.con.execute(SCHEMA)

    def close(self) -> None:
        self.con.close()

    def _insert_df(self, table: str, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        cols = [r[0] for r in self.con.execute(f"DESCRIBE {table}").fetchall()]
        df = df.reindex(columns=cols)
        self.con.register("_tmp_df", df)
        self.con.execute(f"INSERT INTO {table} SELECT * FROM _tmp_df")
        self.con.unregister("_tmp_df")
        return len(df)

    def insert_rows(self, table: str, rows: Iterable[Any]) -> int:
        df = pd.DataFrame([r.as_dict() if hasattr(r, "as_dict") else r for r in rows])
        return self._insert_df(table, df)

    def latest_odds(self) -> pd.DataFrame:
        """Every quote from the most recent pull of each upcoming event.

        One pull returns every book and market at once, so the newest pull is the whole
        board. Keeping older rows would resurrect a spread or total at a point the book has
        since moved off (and a market a book has taken down), and the scan would price
        those dead quotes. row_number drops duplicates if a pull was stored twice.
        """
        return self.con.execute(
            """
            SELECT * EXCLUDE (rn, latest) FROM (
                SELECT *,
                    max(fetched_at) OVER (PARTITION BY event_id) AS latest,
                    row_number() OVER (
                        PARTITION BY event_id, bookmaker, market, outcome, point
                        ORDER BY fetched_at DESC
                    ) AS rn
                FROM odds_snapshots
                WHERE commence_time > now()
            ) WHERE rn = 1 AND fetched_at = latest
            """
        ).df()

    def latest_injuries(self) -> pd.DataFrame:
        return self.con.execute(
            """
            SELECT * EXCLUDE (rn) FROM (
                SELECT *, row_number() OVER (PARTITION BY team, player ORDER BY fetched_at DESC) AS rn
                FROM injuries
            ) WHERE rn = 1
            """
        ).df()

    def line_history(self, event_id: str, bookmaker: str, market: str, outcome: str) -> pd.DataFrame:
        return self.con.execute(
            """
            SELECT fetched_at, point, price FROM odds_snapshots
            WHERE event_id = ? AND bookmaker = ? AND market = ? AND outcome = ?
            ORDER BY fetched_at
            """,
            [event_id, bookmaker, market, outcome],
        ).df()

    def query(self, sql: str, params: list[Any] | None = None) -> pd.DataFrame:
        return self.con.execute(sql, params or []).df()

    # --- bet tracking (used by sportsbet.tracking) ---

    def _columns(self, table: str) -> list[str]:
        return [r[0] for r in self.con.execute(f"DESCRIBE {table}").fetchall()]

    def bets_with_meta(self, open_only: bool = False) -> pd.DataFrame:
        """Every bet joined with its bet_meta row, oldest first."""
        where = "WHERE b.result IS NULL" if open_only else ""
        return self.con.execute(
            f"""
            SELECT b.*, m.* EXCLUDE (bet_id)
            FROM bets b LEFT JOIN bet_meta m USING (bet_id)
            {where}
            ORDER BY b.placed_at, b.bet_id
            """
        ).df()

    def open_bets(self) -> pd.DataFrame:
        """Bets not yet settled (result is null), with their bet_meta fields."""
        return self.bets_with_meta(open_only=True)

    def update_bet(self, bet_id: str, **fields: Any) -> None:
        """Set columns on a bet; each field goes to bets or bet_meta by column name."""
        bet_cols, meta_cols = set(self._columns("bets")), set(self._columns("bet_meta"))
        unknown = set(fields) - bet_cols - meta_cols
        if unknown:
            raise ValueError(f"unknown bet fields: {sorted(unknown)}")
        bet_fields = {k: v for k, v in fields.items() if k in bet_cols and k != "bet_id"}
        meta_fields = {k: v for k, v in fields.items() if k in meta_cols and k not in bet_cols}
        if bet_fields:
            sets = ", ".join(f"{k} = ?" for k in bet_fields)
            self.con.execute(f"UPDATE bets SET {sets} WHERE bet_id = ?", [*bet_fields.values(), bet_id])
        if meta_fields:
            self.con.execute(
                "INSERT INTO bet_meta (bet_id) SELECT ? WHERE NOT EXISTS (SELECT 1 FROM bet_meta WHERE bet_id = ?)",
                [bet_id, bet_id],
            )
            sets = ", ".join(f"{k} = ?" for k in meta_fields)
            self.con.execute(f"UPDATE bet_meta SET {sets} WHERE bet_id = ?", [*meta_fields.values(), bet_id])

    def last_snapshot_before(self, event_id: str, ts: Any) -> pd.DataFrame:
        """All rows of the most recent odds pull for an event fetched strictly before ts."""
        return self.con.execute(
            """
            SELECT * FROM odds_snapshots
            WHERE event_id = ? AND fetched_at = (
                SELECT max(fetched_at) FROM odds_snapshots WHERE event_id = ? AND fetched_at < ?
            )
            """,
            [event_id, event_id, ts],
        ).df()

    def last_quote_before(self, event_id: str, bookmaker: str, market: str, outcome: str, ts: Any) -> pd.DataFrame:
        """A book's most recent quote on one side of a market before ts, at whatever point it had."""
        return self.con.execute(
            """
            SELECT * FROM odds_snapshots
            WHERE event_id = ? AND bookmaker = ? AND market = ? AND outcome = ? AND fetched_at < ?
            ORDER BY fetched_at DESC
            LIMIT 1
            """,
            [event_id, bookmaker, market, outcome, ts],
        ).df()

    def injury_snapshot_times(self, source: str | None = None) -> list[pd.Timestamp]:
        """Distinct fetch times of stored injury snapshots, oldest first."""
        sql = "SELECT DISTINCT fetched_at FROM injuries"
        params: list[Any] = []
        if source is not None:
            sql += " WHERE source = ?"
            params.append(source)
        # via a DataFrame because fetchall() on TIMESTAMPTZ requires pytz
        return self.con.execute(sql + " ORDER BY fetched_at", params).df()["fetched_at"].tolist()

    def injury_snapshot(self, fetched_at: datetime, source: str | None = None) -> pd.DataFrame:
        """Every row of the injury snapshot taken at fetched_at."""
        sql = "SELECT * FROM injuries WHERE fetched_at = ?"
        params: list[Any] = [fetched_at]
        if source is not None:
            sql += " AND source = ?"
            params.append(source)
        return self.con.execute(sql, params).df()

    def previous_injuries(self, source: str | None = None) -> pd.DataFrame:
        """Rows of the snapshot before the most recent one (empty if fewer than two exist).

        Unlike latest_injuries this is a whole snapshot, so a player missing from it was
        not on the report at that time.
        """
        times = self.injury_snapshot_times(source)
        if len(times) < 2:
            return self.con.execute("SELECT * FROM injuries WHERE false").df()
        return self.injury_snapshot(times[-2], source)

    def odds_history(
        self,
        event_id: str | None = None,
        market: str | None = None,
        hours: float | None = 24.0,
        now: datetime | None = None,
    ) -> pd.DataFrame:
        """Odds rows for upcoming events fetched within the last `hours` (None = all), oldest first."""
        now = now or datetime.now(timezone.utc)
        sql = "SELECT * FROM odds_snapshots WHERE commence_time > ?"
        params: list[Any] = [now]
        if hours is not None:
            sql += " AND fetched_at >= ?"
            params.append(now - timedelta(hours=hours))
        if event_id is not None:
            sql += " AND event_id = ?"
            params.append(event_id)
        if market is not None:
            sql += " AND market = ?"
            params.append(market)
        return self.con.execute(sql + " ORDER BY fetched_at", params).df()

    def insert_alerts(self, alerts: Iterable[Any]) -> int:
        return self.insert_rows("alerts", alerts)

    def recent_alert_keys(self, hours: float | None = None) -> set[str]:
        """Keys of alerts already emitted, optionally only those from the last `hours`."""
        sql = "SELECT DISTINCT alert_key FROM alerts"
        params: list[Any] = []
        if hours is not None:
            sql += " WHERE created_at >= ?"
            params.append(datetime.now(timezone.utc) - timedelta(hours=hours))
        return {r[0] for r in self.con.execute(sql, params).fetchall()}
    # --- unattended operation: paid odds pull log and report helpers ---

    def log_pull(
        self,
        pulled_at: datetime,
        slot: str,
        forced: bool,
        n_rows: int,
        credits_last: int | None,
        credits_used: int | None,
        credits_remaining: int | None,
    ) -> None:
        """Record one paid Odds API pull. The weekly cap in schedule.py counts these rows."""
        self._insert_df(
            "pull_log",
            pd.DataFrame(
                [
                    {
                        "pulled_at": pulled_at,
                        "slot": slot,
                        "forced": forced,
                        "n_rows": n_rows,
                        "credits_last": credits_last,
                        "credits_used": credits_used,
                        "credits_remaining": credits_remaining,
                    }
                ]
            ),
        )

    def pull_log(self, since: datetime | None = None) -> pd.DataFrame:
        """Pull log rows, oldest first, with pulled_at as tz-aware UTC."""
        df = self.con.execute(
            "SELECT * FROM pull_log WHERE ? IS NULL OR pulled_at >= ? ORDER BY pulled_at",
            [since, since],
        ).df()
        if not df.empty:
            df["pulled_at"] = pd.to_datetime(df["pulled_at"], utc=True)
        return df

    def pull_times(self, since: datetime | None = None) -> list[datetime]:
        df = self.pull_log(since)
        return [] if df.empty else [t.to_pydatetime() for t in df["pulled_at"]]

    def latest_fetch_time(self, table: str, source: str | None = None) -> datetime | None:
        """Most recent fetched_at in odds_snapshots or injuries (optionally one source), UTC."""
        where = "WHERE source = ?" if source else ""
        df = self.con.execute(
            f"SELECT max(fetched_at) AS t FROM {table} {where}", [source] if source else []
        ).df()
        t = df["t"].iloc[0]
        return None if pd.isna(t) else pd.Timestamp(t).tz_convert("UTC").to_pydatetime()

    def latest_injury_snapshot(self, source: str) -> pd.DataFrame:
        """Only the rows of the newest fetch from one source.

        latest_injuries() keeps a player forever once seen; a live feed drops players
        when they are healthy again, so the newest snapshot alone is the current list.
        DISTINCT guards against a snapshot stored twice double-counting a player.
        """
        return self.con.execute(
            """
            SELECT DISTINCT * FROM injuries
            WHERE source = ? AND fetched_at = (SELECT max(fetched_at) FROM injuries WHERE source = ?)
            """,
            [source, source],
        ).df()
