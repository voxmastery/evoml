"""Append-only SQLite journal of arena decisions.

Rows are inserted and read. The schema refuses UPDATE and DELETE so a
process with this connection cannot rewrite history.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any


_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    arena TEXT NOT NULL,
    generation INTEGER NOT NULL,
    champion TEXT NOT NULL,
    promoted INTEGER NOT NULL,
    succession TEXT NOT NULL,
    genome TEXT NOT NULL,
    scores TEXT NOT NULL,
    record TEXT NOT NULL
);
"""

_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS audit_no_update
BEFORE UPDATE ON audit
BEGIN
    SELECT RAISE(ABORT, 'audit log is append-only');
END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete
BEFORE DELETE ON audit
BEGIN
    SELECT RAISE(ABORT, 'audit log is append-only');
END;
"""


class AuditLog:
    def __init__(self, path: str | Path):
        self.path = path
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.executescript(_TRIGGERS)
        self._conn.commit()

    def append(self, record: dict[str, Any]) -> int:
        """Insert one decision. Returns the new row id. Never updates."""
        payload = {k: v for k, v in record.items() if k != "id"}
        now = float(payload.get("ts", time.time()))
        cur = self._conn.execute(
            "INSERT INTO audit (ts, arena, generation, champion, promoted, "
            "succession, genome, scores, record) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                now,
                str(payload.get("arena", "")),
                int(payload.get("generation", 0)),
                str(payload.get("champion", "")),
                1 if payload.get("promoted") else 0,
                str(payload.get("succession", "")),
                json.dumps(payload.get("genome", {}), sort_keys=True, default=str),
                json.dumps(payload.get("scores", {}), sort_keys=True, default=str),
                json.dumps(payload, sort_keys=True, default=str),
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def read(self, limit: int | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM audit ORDER BY id ASC"
        if limit is not None:
            sql += " LIMIT ?"
            rows = self._conn.execute(sql, (int(limit),)).fetchall()
        else:
            rows = self._conn.execute(sql).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["record"] = json.loads(item["record"])
            item["scores"] = json.loads(item["scores"])
            item["genome"] = json.loads(item["genome"])
            out.append(item)
        return out

    def close(self) -> None:
        self._conn.close()
