"""DuckDB storage for odds snapshots, injuries, scan candidates and tracked bets."""

from __future__ import annotations

from collections.abc import Iterable
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
