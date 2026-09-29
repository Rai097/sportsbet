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
