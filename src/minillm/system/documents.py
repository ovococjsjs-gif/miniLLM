"""Small, auditable SQLite FTS retrieval store for local documents."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

_INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(all\s+)?previous", re.IGNORECASE),
    re.compile(r"system\s+prompt", re.IGNORECASE),
    re.compile(r"you\s+are\s+(now|chatgpt|an?\s+assistant)", re.IGNORECASE),
    re.compile(r"do\s+not\s+follow", re.IGNORECASE),
    re.compile(r"<\/?(system|assistant|tool)", re.IGNORECASE),
)


@dataclass(frozen=True)
class DocumentChunk:
    citation_id: str
    document_id: int
    title: str
    source: str
    text: str
    score: float
    injection_warning: bool


class DocumentStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.connection = sqlite3.connect(str(path))
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                source TEXT NOT NULL,
                license TEXT NOT NULL,
                content_hash TEXT NOT NULL UNIQUE,
                created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL,
                text TEXT NOT NULL,
                injection_warning INTEGER NOT NULL DEFAULT 0,
                UNIQUE(document_id, ordinal)
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text, content='chunks', content_rowid='id'
            );
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
            END;
            PRAGMA foreign_keys = ON;
            """
        )

    @staticmethod
    def _has_injection(text: str) -> bool:
        return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)

    @staticmethod
    def chunk_text(
        text: str, *, target_characters: int = 1200, overlap: int = 160
    ) -> list[str]:
        if target_characters < 128 or not 0 <= overlap < target_characters:
            raise ValueError("invalid chunking parameters")
        paragraphs = [
            part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()
        ]
        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs:
            if current and len(current) + len(paragraph) + 2 > target_characters:
                chunks.append(current)
                current = current[-overlap:] + "\n\n" + paragraph
            else:
                current = paragraph if not current else current + "\n\n" + paragraph
        if current:
            chunks.append(current)
        return chunks

    def add_document(
        self,
        *,
        title: str,
        source: str,
        license: str,
        text: str,
        content_hash: str,
        created_at: str | None = None,
    ) -> int:
        chunks = self.chunk_text(text)
        if not chunks:
            raise ValueError("document text is empty")
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO documents(title,source,license,content_hash,created_at) VALUES(?,?,?,?,?)",
                (title, source, license, content_hash, created_at),
            )
            document_id = int(cursor.lastrowid)
            self.connection.executemany(
                "INSERT INTO chunks(document_id,ordinal,text,injection_warning) VALUES(?,?,?,?)",
                [
                    (document_id, ordinal, chunk, int(self._has_injection(chunk)))
                    for ordinal, chunk in enumerate(chunks)
                ],
            )
        return document_id

    def search(self, query: str, *, limit: int = 8) -> list[DocumentChunk]:
        tokens = [token.replace('"', '""') for token in query.split() if token]
        if not tokens or limit < 1:
            return []
        match = " OR ".join(f'"{token}"' for token in tokens)
        rows = self.connection.execute(
            """
            SELECT chunks.id AS chunk_id, chunks.document_id, chunks.text,
                   chunks.injection_warning, documents.title, documents.source,
                   bm25(chunks_fts) AS rank
            FROM chunks_fts
            JOIN chunks ON chunks.id = chunks_fts.rowid
            JOIN documents ON documents.id = chunks.document_id
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (match, limit),
        ).fetchall()
        return [
            DocumentChunk(
                citation_id=f"doc:{row['document_id']}:chunk:{row['chunk_id']}",
                document_id=row["document_id"],
                title=row["title"],
                source=row["source"],
                text=row["text"],
                score=-float(row["rank"]),
                injection_warning=bool(row["injection_warning"]),
            )
            for row in rows
        ]

    def close(self) -> None:
        self.connection.close()
