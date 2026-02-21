"""
database.py — SQLite layer for storing indexed person data.
"""
import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join("models", "faces.db")


def get_connection() -> sqlite3.Connection:
    """Create a database connection with WAL mode enabled for better performance."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db():
    """Create the table if it does not exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT    NOT NULL,
                image_path TEXT    NOT NULL,
                created_at TEXT    NOT NULL
            )
        """)
        conn.commit()


def add_person(name: str, image_path: str) -> int:
    """
    Add a new person and return the auto-generated ID.
    This ID will be used to link the embedding in FAISS.
    """
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO persons (name, image_path, created_at) VALUES (?, ?, ?)",
            (name, image_path, now),
        )
        conn.commit()
        return cursor.lastrowid


def get_person_by_id(person_id: int) -> dict | None:
    """Return a single person's data by ID."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, name, image_path, created_at FROM persons WHERE id = ?",
            (person_id,),
        ).fetchone()
        return dict(row) if row else None


def get_persons_by_ids(ids: list[int]) -> dict[int, dict]:
    """Return data for multiple persons by a list of IDs — single query instead of N."""
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT id, name, image_path, created_at FROM persons WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        return {row["id"]: dict(row) for row in rows}


def get_all_persons() -> list[dict]:
    """Return all indexed persons."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, name, image_path, created_at FROM persons ORDER BY id"
        ).fetchall()
        return [dict(row) for row in rows]


def get_persons_paginated(skip: int = 0, limit: int = 200, order: str = "desc") -> dict:
    """Return paginated persons with total count. order: 'desc'=newest, 'asc'=oldest."""
    direction = "DESC" if order == "desc" else "ASC"
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
        rows = conn.execute(
            f"SELECT id, name, image_path, created_at FROM persons "
            f"ORDER BY id {direction} LIMIT ? OFFSET ?",
            (limit, skip),
        ).fetchall()
        return {"total": total, "persons": [dict(row) for row in rows]}


def update_person_name(person_id: int, name: str):
    """Update an existing person's name."""
    with get_connection() as conn:
        conn.execute("UPDATE persons SET name = ? WHERE id = ?", (name, person_id))
        conn.commit()


def delete_person(person_id: int) -> bool:
    """Delete a person from the database. Returns True if deletion succeeded."""
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM persons WHERE id = ?", (person_id,))
        conn.commit()
        return cursor.rowcount > 0
