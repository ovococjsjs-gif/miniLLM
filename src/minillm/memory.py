"""Auditable, temporal episodic memory backed by SQLite.

This store is intentionally separate from learned Engram memory. Facts can be inspected,
versioned, filtered by privacy class, and deleted without retraining model weights.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

_PRIVACY = {"public", "private", "sensitive"}


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class MemoryFact:
    subject: str
    predicate: str
    object: str
    source_turn: str
    confidence: float = 1.0
    privacy_class: str = "private"
    valid_from: str | None = None
    valid_to: str | None = None
    id: int | None = None
    created_at: str | None = None
    last_confirmed: str | None = None
    superseded_by: int | None = None

    def validate(self) -> MemoryFact:
        for name in ("subject", "predicate", "object", "source_turn"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.privacy_class not in _PRIVACY:
            raise ValueError(f"privacy_class must be one of {sorted(_PRIVACY)}")
        return self


class EpisodicMemoryStore:
    """Versioned facts with exact, temporal, and lexical retrieval."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.connection = sqlite3.connect(str(path))
        self.connection.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                valid_from TEXT,
                valid_to TEXT,
                source_turn TEXT NOT NULL,
                confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
                privacy_class TEXT NOT NULL CHECK(privacy_class IN ('public','private','sensitive')),
                created_at TEXT NOT NULL,
                last_confirmed TEXT NOT NULL,
                superseded_by INTEGER REFERENCES facts(id) ON DELETE SET NULL
            );
            CREATE INDEX IF NOT EXISTS idx_fact_relation
                ON facts(subject, predicate, valid_from, valid_to);
            CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
                subject, predicate, object, content='facts', content_rowid='id'
            );
            CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
                INSERT INTO facts_fts(rowid, subject, predicate, object)
                VALUES (new.id, new.subject, new.predicate, new.object);
            END;
            CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
                INSERT INTO facts_fts(facts_fts, rowid, subject, predicate, object)
                VALUES ('delete', old.id, old.subject, old.predicate, old.object);
            END;
            CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
                INSERT INTO facts_fts(facts_fts, rowid, subject, predicate, object)
                VALUES ('delete', old.id, old.subject, old.predicate, old.object);
                INSERT INTO facts_fts(rowid, subject, predicate, object)
                VALUES (new.id, new.subject, new.predicate, new.object);
            END;
            """
        )
        self.connection.commit()

    @staticmethod
    def _from_row(row: sqlite3.Row) -> MemoryFact:
        return MemoryFact(**dict(row))

    def add(self, fact: MemoryFact) -> MemoryFact:
        fact.validate()
        now = _now()
        cursor = self.connection.execute(
            """
            INSERT INTO facts(
                subject, predicate, object, valid_from, valid_to, source_turn,
                confidence, privacy_class, created_at, last_confirmed, superseded_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fact.subject,
                fact.predicate,
                fact.object,
                fact.valid_from,
                fact.valid_to,
                fact.source_turn,
                fact.confidence,
                fact.privacy_class,
                fact.created_at or now,
                fact.last_confirmed or now,
                fact.superseded_by,
            ),
        )
        self.connection.commit()
        return self.get(int(cursor.lastrowid))

    def get(self, fact_id: int) -> MemoryFact:
        row = self.connection.execute(
            "SELECT * FROM facts WHERE id = ?", (fact_id,)
        ).fetchone()
        if row is None:
            raise KeyError(fact_id)
        return self._from_row(row)

    def supersede(self, previous_id: int, replacement: MemoryFact) -> MemoryFact:
        """Add a new version and close the old fact in one transaction."""

        replacement.validate()
        previous = self.get(previous_id)
        if (previous.subject, previous.predicate) != (
            replacement.subject,
            replacement.predicate,
        ):
            raise ValueError(
                "replacement must describe the same subject/predicate relation"
            )
        now = _now()
        with self.connection:
            cursor = self.connection.execute(
                """
                INSERT INTO facts(
                    subject, predicate, object, valid_from, valid_to, source_turn,
                    confidence, privacy_class, created_at, last_confirmed, superseded_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    replacement.subject,
                    replacement.predicate,
                    replacement.object,
                    replacement.valid_from,
                    replacement.valid_to,
                    replacement.source_turn,
                    replacement.confidence,
                    replacement.privacy_class,
                    replacement.created_at or now,
                    replacement.last_confirmed or now,
                    replacement.superseded_by,
                ),
            )
            inserted_id = int(cursor.lastrowid)
            boundary = replacement.valid_from or replacement.created_at or now
            self.connection.execute(
                "UPDATE facts SET valid_to = ?, superseded_by = ? WHERE id = ?",
                (boundary, inserted_id, previous_id),
            )
        return self.get(inserted_id)

    def confirm(
        self, fact_id: int, *, at: str | None = None, confidence: float | None = None
    ) -> None:
        if confidence is not None and not 0 <= confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")
        if confidence is None:
            cursor = self.connection.execute(
                "UPDATE facts SET last_confirmed = ? WHERE id = ?",
                (at or _now(), fact_id),
            )
        else:
            cursor = self.connection.execute(
                "UPDATE facts SET last_confirmed = ?, confidence = ? WHERE id = ?",
                (at or _now(), confidence, fact_id),
            )
        if cursor.rowcount != 1:
            raise KeyError(fact_id)
        self.connection.commit()

    def relation(
        self,
        subject: str,
        predicate: str,
        *,
        at: str | None = None,
        allowed_privacy: Iterable[str] = ("public", "private"),
    ) -> list[MemoryFact]:
        allowed = tuple(allowed_privacy)
        if not allowed or set(allowed) - _PRIVACY:
            raise ValueError("invalid privacy filter")
        placeholders = ",".join("?" for _ in allowed)
        query = f"""
            SELECT * FROM facts
            WHERE subject = ? AND predicate = ?
              AND privacy_class IN ({placeholders})
        """
        parameters: list[object] = [subject, predicate, *allowed]
        if at is None:
            query += " AND superseded_by IS NULL AND valid_to IS NULL"
        else:
            query += " AND (valid_from IS NULL OR valid_from <= ?) AND (valid_to IS NULL OR valid_to > ?)"
            parameters.extend((at, at))
        query += " ORDER BY confidence DESC, last_confirmed DESC"
        return [
            self._from_row(row) for row in self.connection.execute(query, parameters)
        ]

    def search(
        self,
        text: str,
        *,
        limit: int = 8,
        allowed_privacy: Iterable[str] = ("public", "private"),
    ) -> list[MemoryFact]:
        if not text.strip() or limit < 1:
            return []
        allowed = tuple(allowed_privacy)
        if not allowed or set(allowed) - _PRIVACY:
            raise ValueError("invalid privacy filter")
        placeholders = ",".join("?" for _ in allowed)
        # Quote lexical tokens so punctuation from user input cannot become FTS syntax.
        tokens = [token.replace('"', '""') for token in text.split() if token]
        match = " OR ".join(f'"{token}"' for token in tokens)
        query = f"""
            SELECT facts.* FROM facts_fts
            JOIN facts ON facts.id = facts_fts.rowid
            WHERE facts_fts MATCH ?
              AND facts.privacy_class IN ({placeholders})
            ORDER BY bm25(facts_fts), facts.confidence DESC
            LIMIT ?
        """
        rows = self.connection.execute(query, (match, *allowed, limit)).fetchall()
        return [self._from_row(row) for row in rows]

    def delete(self, fact_id: int) -> None:
        cursor = self.connection.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
        if cursor.rowcount != 1:
            raise KeyError(fact_id)
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
