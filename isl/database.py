"""SQLite storage – implements the ER diagram of the synopsis.

USER -> users, SESSION -> sessions, GESTURE -> gestures,
TRANSLATION_HISTORY -> translation_history, DATASET_SAMPLE -> dataset_samples
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(user_id),
    start_time  TEXT NOT NULL,
    end_time    TEXT
);
CREATE TABLE IF NOT EXISTS gestures (
    gesture_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    gesture_name  TEXT NOT NULL UNIQUE,
    meaning       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS translation_history (
    history_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       INTEGER NOT NULL REFERENCES sessions(session_id),
    gesture_id       INTEGER NOT NULL REFERENCES gestures(gesture_id),
    translated_text  TEXT NOT NULL,
    confidence       REAL,
    date_time        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dataset_samples (
    sample_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    gesture_id         INTEGER NOT NULL REFERENCES gestures(gesture_id),
    landmark_features  TEXT NOT NULL,
    label              TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def get_conn():
    """Open a connection, commit on success, always close."""
    conn = sqlite3.connect(config.DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def seed_gestures(gestures: dict) -> None:
    """Make sure every supported gesture exists in the GESTURE table."""
    with get_conn() as conn:
        conn.executemany(
            "INSERT INTO gestures(gesture_name, meaning) VALUES (?, ?) "
            "ON CONFLICT(gesture_name) DO UPDATE SET meaning = excluded.meaning",
            list(gestures.items()),
        )


def _gesture_id(conn, name: str) -> int:
    row = conn.execute("SELECT gesture_id FROM gestures WHERE gesture_name = ?", (name,)).fetchone()
    if row:
        return row["gesture_id"]
    cur = conn.execute("INSERT INTO gestures(gesture_name, meaning) VALUES (?, ?)",
                       (name, config.GESTURES.get(name, name)))
    return cur.lastrowid


def get_or_create_user(name: str) -> int:
    with get_conn() as conn:
        row = conn.execute("SELECT user_id FROM users WHERE name = ?", (name,)).fetchone()
        if row:
            return row["user_id"]
        cur = conn.execute("INSERT INTO users(name, created_at) VALUES (?, ?)", (name, _now()))
        return cur.lastrowid


def start_session(user_id: int) -> int:
    with get_conn() as conn:
        cur = conn.execute("INSERT INTO sessions(user_id, start_time) VALUES (?, ?)", (user_id, _now()))
        return cur.lastrowid


def end_session(session_id: int) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE sessions SET end_time = ? WHERE session_id = ? AND end_time IS NULL",
                     (_now(), session_id))


def add_translation(session_id: int, gesture_name: str, translated_text: str, confidence: float) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO translation_history(session_id, gesture_id, translated_text, confidence, date_time) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, _gesture_id(conn, gesture_name), translated_text, float(confidence), _now()),
        )


def get_history(limit: int = 500, session_id: int | None = None) -> pd.DataFrame:
    sql = """
        SELECT h.history_id AS "ID", h.date_time AS "Date/Time", u.name AS "User",
               h.session_id AS "Session", g.gesture_name AS "Gesture",
               h.translated_text AS "Translated text",
               ROUND(h.confidence * 100, 1) AS "Confidence %"
        FROM translation_history h
        JOIN gestures g ON g.gesture_id = h.gesture_id
        JOIN sessions s ON s.session_id = h.session_id
        JOIN users u    ON u.user_id = s.user_id
    """
    params: list = []
    if session_id is not None:
        sql += " WHERE h.session_id = ?"
        params.append(session_id)
    sql += " ORDER BY h.history_id DESC LIMIT ?"
    params.append(int(limit))
    with get_conn() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def clear_history() -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM translation_history")


def add_dataset_samples(label: str, feature_rows) -> None:
    """Store many landmark samples of one gesture (one insert for all rows)."""
    with get_conn() as conn:
        gid = _gesture_id(conn, label)
        conn.executemany(
            "INSERT INTO dataset_samples(gesture_id, landmark_features, label) VALUES (?, ?, ?)",
            [(gid, json.dumps([round(float(v), 6) for v in row]), label) for row in feature_rows],
        )


def rename_gesture(old: str, new: str) -> None:
    """Rename a sign everywhere in the database (history keeps pointing at it)."""
    with get_conn() as conn:
        conn.execute("UPDATE gestures SET gesture_name = ?, meaning = ? WHERE gesture_name = ?",
                     (new, config.GESTURES.get(new, new), old))
        conn.execute("UPDATE dataset_samples SET label = ? WHERE label = ?", (new, old))


def delete_dataset_samples(label: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM dataset_samples WHERE label = ?", (label,))


def count_dataset_samples() -> int:
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM dataset_samples").fetchone()[0]
