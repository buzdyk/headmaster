import sqlite3
import struct
from pathlib import Path

DB_NAME = "headmaster.db"

SCHEMA = """\
CREATE TABLE IF NOT EXISTS models (
    id        INTEGER PRIMARY KEY,
    name      TEXT UNIQUE NOT NULL,
    path      TEXT NOT NULL,
    embed_dim INTEGER NOT NULL,
    active    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS embeddings (
    hash     TEXT NOT NULL,
    model_id INTEGER NOT NULL REFERENCES models(id),
    vector   BLOB NOT NULL,
    PRIMARY KEY (hash, model_id)
);
"""


def _connect(workspace: Path) -> sqlite3.Connection:
    db_path = workspace / DB_NAME
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(workspace: Path) -> None:
    conn = _connect(workspace)
    conn.executescript(SCHEMA)
    conn.close()


# ── Model registry ──────────────────────────────────────────────


def model_add(workspace: Path, name: str, path: str, embed_dim: int) -> None:
    conn = _connect(workspace)
    try:
        conn.execute(
            "INSERT INTO models (name, path, embed_dim) VALUES (?, ?, ?)",
            (name, path, embed_dim),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        raise SystemExit(f"error: model '{name}' already exists")
    finally:
        conn.close()


def model_list(workspace: Path) -> list[dict]:
    conn = _connect(workspace)
    rows = conn.execute(
        "SELECT name, path, embed_dim, active FROM models ORDER BY id"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def model_activate(workspace: Path, name: str) -> None:
    conn = _connect(workspace)
    row = conn.execute("SELECT id FROM models WHERE name = ?", (name,)).fetchone()
    if row is None:
        conn.close()
        raise SystemExit(f"error: model '{name}' not found")
    conn.execute("UPDATE models SET active = 0")
    conn.execute("UPDATE models SET active = 1 WHERE name = ?", (name,))
    conn.commit()
    conn.close()


def model_remove(workspace: Path, name: str) -> None:
    conn = _connect(workspace)
    row = conn.execute("SELECT id FROM models WHERE name = ?", (name,)).fetchone()
    if row is None:
        conn.close()
        raise SystemExit(f"error: model '{name}' not found")
    model_id = row["id"]
    conn.execute("DELETE FROM embeddings WHERE model_id = ?", (model_id,))
    conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
    conn.commit()
    conn.close()


def get_active_model(workspace: Path) -> dict | None:
    conn = _connect(workspace)
    row = conn.execute(
        "SELECT id, name, path, embed_dim FROM models WHERE active = 1"
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# ── Embedding cache ─────────────────────────────────────────────


def get_embedding(workspace: Path, hash: str, model_id: int) -> bytes | None:
    conn = _connect(workspace)
    row = conn.execute(
        "SELECT vector FROM embeddings WHERE hash = ? AND model_id = ?",
        (hash, model_id),
    ).fetchone()
    conn.close()
    return row["vector"] if row else None


def put_embedding(workspace: Path, hash: str, model_id: int, vector: bytes) -> None:
    conn = _connect(workspace)
    conn.execute(
        "INSERT OR REPLACE INTO embeddings (hash, model_id, vector) VALUES (?, ?, ?)",
        (hash, model_id, vector),
    )
    conn.commit()
    conn.close()


def get_cached_hashes(workspace: Path, model_id: int) -> set[str]:
    conn = _connect(workspace)
    rows = conn.execute(
        "SELECT hash FROM embeddings WHERE model_id = ?", (model_id,)
    ).fetchall()
    conn.close()
    return {r["hash"] for r in rows}
