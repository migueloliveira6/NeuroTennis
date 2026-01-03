import sqlite3
import pandas as pd
from db import get_connection, load_matches_for_training


def _create_inmemory_matches_db():
    conn = sqlite3.connect(':memory:')
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE matches (
            match_id INTEGER PRIMARY KEY AUTOINCREMENT,
            tourney_date DATE,
            winner_name TEXT,
            loser_name TEXT,
            surface TEXT,
            winner_elo_after REAL,
            loser_elo_after REAL
        )
    ''')
    # Insert sample rows
    rows = [
        ('2025-01-01', 'A', 'B', 'Hard', 1520, 1480),
        ('2025-02-01', 'C', 'D', 'Clay', 1510, 1490),
        ('2025-03-01', 'E', 'F', 'Grass', 1530, 1470),
    ]
    cur.executemany('INSERT INTO matches (tourney_date, winner_name, loser_name, surface, winner_elo_after, loser_elo_after) VALUES (?, ?, ?, ?, ?, ?)', rows)
    conn.commit()
    return conn


def test_load_matches_for_training_all():
    conn = _create_inmemory_matches_db()
    df = load_matches_for_training(conn)
    assert len(df) == 3
    assert 'tourney_date' in df.columns
    conn.close()


def test_load_matches_for_training_with_date_range():
    conn = _create_inmemory_matches_db()
    df = load_matches_for_training(conn, start_date='2025-02-01', end_date='2025-04-01')
    assert len(df) == 2
    assert set(df['surface']) == {'Clay', 'Grass'}
    conn.close()
