from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .schema import TextDocument


class DocumentStore:
    def __init__(self, raw_root: Path, state_path: Path):
        self.raw_root = Path(raw_root)
        self.state_path = Path(state_path)
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.state_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                document_type TEXT NOT NULL,
                source TEXT NOT NULL,
                published_at TEXT,
                first_seen_at TEXT NOT NULL,
                raw_path TEXT NOT NULL,
                UNIQUE(source, content_hash)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS source_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL,
                fetched INTEGER DEFAULT 0,
                inserted INTEGER DEFAULT 0,
                error TEXT
            )
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "DocumentStore":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    @staticmethod
    def _date_parts(document: TextDocument) -> tuple[str, str, str]:
        value = document.published_at or document.first_seen_at
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            parsed = datetime.now(timezone.utc)
        return parsed.strftime("%Y"), parsed.strftime("%m"), parsed.strftime("%d")

    def put(self, document: TextDocument) -> tuple[bool, Path]:
        existing = self.conn.execute(
            "SELECT raw_path FROM documents WHERE document_id=? OR (source=? AND content_hash=?)",
            (document.document_id, document.source, document.content_hash),
        ).fetchone()
        if existing:
            return False, self.raw_root / existing[0]

        year, month, day = self._date_parts(document)
        relative = Path(document.document_type) / year / month / day / f"{document.document_id}.json"
        output = self.raw_root / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(document.to_json(), encoding="utf-8")
        os.replace(temporary, output)
        try:
            self.conn.execute(
                """
                INSERT INTO documents
                    (document_id, content_hash, document_type, source, published_at,
                     first_seen_at, raw_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.document_id,
                    document.content_hash,
                    document.document_type,
                    document.source,
                    document.published_at,
                    document.first_seen_at,
                    relative.as_posix(),
                ),
            )
            self.conn.commit()
        except Exception:
            output.unlink(missing_ok=True)
            raise
        return True, output

    def put_many(self, documents: Iterable[TextDocument]) -> tuple[int, int]:
        fetched = inserted = 0
        for document in documents:
            fetched += 1
            created, _ = self.put(document)
            inserted += int(created)
        return fetched, inserted

    def start_run(self, source: str, started_at: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO source_runs(source, started_at, status) VALUES (?, ?, 'running')",
            (source, started_at),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def finish_run(
        self, run_id: int, finished_at: str, status: str, fetched: int, inserted: int,
        error: str = "",
    ) -> None:
        self.conn.execute(
            """
            UPDATE source_runs SET finished_at=?, status=?, fetched=?, inserted=?, error=?
            WHERE run_id=?
            """,
            (finished_at, status, fetched, inserted, error[:2000], run_id),
        )
        self.conn.commit()

    def stats(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT document_type, COUNT(*) FROM documents GROUP BY document_type"
        ).fetchall()
        return {str(kind): int(count) for kind, count in rows}

