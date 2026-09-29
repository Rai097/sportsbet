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
