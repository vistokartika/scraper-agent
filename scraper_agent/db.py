from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional

DB_PATH = os.environ.get("SCRAPER_AGENT_DB", os.path.abspath("sessions.sqlite"))


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                session_id TEXT PRIMARY KEY,
                schema_json TEXT,
                items_json TEXT,
                visited_count INTEGER,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def ensure_session(session_id: str) -> None:
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("INSERT OR IGNORE INTO sessions(session_id) VALUES(?)", (session_id,))
        conn.commit()
    finally:
        conn.close()


def add_message(session_id: str, role: str, content: str) -> None:
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO messages(session_id, role, content) VALUES(?,?,?)",
            (session_id, role, content),
        )
        conn.commit()
    finally:
        conn.close()


def list_messages(session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT role, content, created_at FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def save_results(session_id: str, schema: Dict[str, Any], items: List[Dict[str, Any]], visited_count: int) -> None:
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO results(session_id, schema_json, items_json, visited_count)
            VALUES(?,?,?,?)
            ON CONFLICT(session_id) DO UPDATE SET
                schema_json=excluded.schema_json,
                items_json=excluded.items_json,
                visited_count=excluded.visited_count,
                updated_at=CURRENT_TIMESTAMP
            """,
            (session_id, json.dumps(schema), json.dumps(items), visited_count),
        )
        conn.commit()
    finally:
        conn.close()


def get_results(session_id: str) -> Optional[Dict[str, Any]]:
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT schema_json, items_json, visited_count, updated_at FROM results WHERE session_id=?",
            (session_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        schema = json.loads(row[0]) if row[0] else {}
        items = json.loads(row[1]) if row[1] else []
        return {
            "schema": schema,
            "items": items,
            "visited_count": row[2],
            "updated_at": row[3],
        }
    finally:
        conn.close()
