import sqlite3
from pathlib import Path
from typing import Any, Dict, List

DB_PATH = Path(__file__).resolve().parent.parent / "bets.db"


def _get_connection():
    """Return a connection to the bets database."""
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    """Create the bets table if it does not already exist."""
    with _get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                league TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                bet_type TEXT NOT NULL,
                odds REAL NOT NULL,
                stake REAL NOT NULL,
                result TEXT,
                profit REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def insert_bet(
    league: str,
    home_team: str,
    away_team: str,
    bet_type: str,
    odds: float,
    stake: float,
    result: str | None = None,
    profit: float | None = None,
    created_at: str | None = None,
) -> int:
    """Insert a new bet and return its id."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO bets (
                league, home_team, away_team, bet_type, odds, stake, result, profit, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))
            """,
            (
                league,
                home_team,
                away_team,
                bet_type,
                odds,
                stake,
                result,
                profit,
                created_at,
            ),
        )
        conn.commit()
        return cursor.lastrowid


def update_bet(bet_id: int, **fields: Any) -> None:
    """Update a bet's fields by id."""
    if not fields:
        return
    assignments = ", ".join(f"{key} = ?" for key in fields)
    values = list(fields.values()) + [bet_id]
    with _get_connection() as conn:
        conn.execute(f"UPDATE bets SET {assignments} WHERE id = ?", values)
        conn.commit()


def fetch_bets() -> List[Dict[str, Any]]:
    """Return all bets as a list of dicts ordered by creation time."""
    with _get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM bets ORDER BY created_at DESC")
        return [dict(row) for row in cursor.fetchall()]


def compute_stats() -> Dict[str, float]:
    """Compute ROI and win rate for all recorded bets."""
    with _get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT
                SUM(stake) AS total_stake,
                SUM(profit) AS total_profit,
                SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) AS wins,
                COUNT(*) AS total_bets
            FROM bets
            """
        )
        row = cursor.fetchone()
        total_stake = row[0] or 0.0
        total_profit = row[1] or 0.0
        wins = row[2] or 0
        total_bets = row[3] or 0

    roi = total_profit / total_stake if total_stake else 0.0
    win_rate = wins / total_bets if total_bets else 0.0
    return {"roi": roi, "win_rate": win_rate}
