"""DuckDB storage for odds snapshots, injuries, scan candidates and tracked bets."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
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
        """Most recent price per (event, book, market, outcome, point) across all snapshots."""
        return self.con.execute(
            """
            SELECT * EXCLUDE (rn) FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY event_id, bookmaker, market, outcome, point
                    ORDER BY fetched_at DESC
                ) AS rn
                FROM odds_snapshots
                WHERE commence_time > now()
            ) WHERE rn = 1
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
