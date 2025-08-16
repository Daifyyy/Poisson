import math
import sqlite3

import pytest

from utils import bet_db


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "bets.db"
    monkeypatch.setattr(bet_db, "DB_PATH", db_path)
    return db_path


def test_init_db_creates_table(temp_db):
    assert not temp_db.exists()
    bet_db.init_db()
    assert temp_db.exists()
    with sqlite3.connect(temp_db) as conn:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='bets'"
        )
        assert cursor.fetchone() is not None


def test_insert_and_fetch(temp_db):
    bet_db.init_db()
    bet_id = bet_db.insert_bet(
        league="League",
        home_team="A",
        away_team="B",
        bet_type="1x2",
        odds=2.0,
        stake=10.0,
        created_at="2024-01-01 00:00:00",
    )
    bets = bet_db.fetch_bets()
    assert len(bets) == 1
    bet = bets[0]
    assert bet["id"] == bet_id
    assert bet["league"] == "League"
    assert bet["home_team"] == "A"
    assert bet["away_team"] == "B"
    assert bet["bet_type"] == "1x2"
    assert bet["odds"] == 2.0
    assert bet["stake"] == 10.0
    assert bet["result"] is None
    assert bet["profit"] is None


def test_update_bet(temp_db):
    bet_db.init_db()
    bet_id = bet_db.insert_bet(
        league="League",
        home_team="A",
        away_team="B",
        bet_type="1x2",
        odds=2.0,
        stake=10.0,
    )
    bet_db.update_bet(bet_id, result="win", profit=5.0)
    bet = bet_db.fetch_bets()[0]
    assert bet["result"] == "win"
    assert bet["profit"] == 5.0


def test_compute_stats(temp_db):
    bet_db.init_db()
    bet_db.insert_bet(
        league="L",
        home_team="A",
        away_team="B",
        bet_type="1x2",
        odds=2.0,
        stake=10.0,
        result="win",
        profit=5.0,
    )
    bet_db.insert_bet(
        league="L",
        home_team="C",
        away_team="D",
        bet_type="1x2",
        odds=1.5,
        stake=20.0,
        result="loss",
        profit=-20.0,
    )
    stats = bet_db.compute_stats()
    assert math.isclose(stats["roi"], -0.5)
    assert math.isclose(stats["win_rate"], 0.5)
