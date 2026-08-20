"""Deterministic gzip shards and disk-backed deduplication for larger corpora."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import sqlite3
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, TextIO

from .corpus import (
    ContaminationIndex,
    CorpusDocument,
    Rejection,
    assess_quality,
    hamming_distance,
    normalize_text,
    simhash64,
    stable_split,
)
from .data_policy import DataPolicy, SourceRegistry


@dataclass(frozen=True)
class ShardInfo:
    split: str
    path: str
    documents: int
    utf8_bytes: int
    compressed_bytes: int
    sha256: str


@dataclass
class _OpenShard:
    split: str
    index: int
    temporary_path: Path
    final_path: Path
    raw: Any
    text: TextIO
    documents: int = 0
    utf8_bytes: int = 0


class DeterministicShardWriter:
    """Write stable JSONL.GZ shards without embedding wall-clock timestamps."""

    def __init__(self, root: str | Path, *, max_uncompressed_bytes: int) -> None:
        if max_uncompressed_bytes < 1:
            raise ValueError("max shard size must be positive")
        self.root = Path(root)
        self.max_uncompressed_bytes = max_uncompressed_bytes
        self.open_shards: dict[str, _OpenShard] = {}
        self.next_indices: Counter[str] = Counter()
        self.finished: list[ShardInfo] = []

    def _open(self, split: str) -> _OpenShard:
        directory = self.root / split
        directory.mkdir(parents=True, exist_ok=True)
        index = self.next_indices[split]
        self.next_indices[split] += 1
        final_path = directory / f"part-{index:05d}.jsonl.gz"
        temporary_path = final_path.with_suffix(final_path.suffix + ".tmp")
        if final_path.exists() or temporary_path.exists():
            raise FileExistsError(f"refusing to overwrite corpus shard {final_path}")
        raw = temporary_path.open("wb")
        compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
        text = io.TextIOWrapper(compressed, encoding="utf-8", newline="\n")
        opened = _OpenShard(split, index, temporary_path, final_path, raw, text)
        self.open_shards[split] = opened
        return opened

    def _close(self, split: str) -> None:
        opened = self.open_shards.pop(split)
        opened.text.flush()
        opened.text.close()
        if not opened.raw.closed:
            opened.raw.close()
        os.replace(opened.temporary_path, opened.final_path)
        digest = hashlib.sha256()
        with opened.final_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        self.finished.append(
            ShardInfo(
                split=split,
                path=str(opened.final_path.relative_to(self.root)),
                documents=opened.documents,
                utf8_bytes=opened.utf8_bytes,
                compressed_bytes=opened.final_path.stat().st_size,
                sha256=digest.hexdigest(),
            )
        )

    def write(self, split: str, document: CorpusDocument) -> None:
        record = {**asdict(document), "_content_sha256": document.sha256}
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        size = len(line.encode("utf-8"))
        opened = self.open_shards.get(split)
        if (
            opened is not None
            and opened.documents
            and opened.utf8_bytes + size > self.max_uncompressed_bytes
        ):
            self._close(split)
            opened = None
        if opened is None:
            opened = self._open(split)
        opened.text.write(line)
        opened.documents += 1
        opened.utf8_bytes += size

    def close(self) -> tuple[ShardInfo, ...]:
        for split in sorted(self.open_shards):
            self._close(split)
        return tuple(sorted(self.finished, key=lambda item: item.path))


class SQLiteDedupIndex:
    """Exact and LSH-assisted near-dedup index that does not grow in Python RAM."""

    def __init__(self, path: str | Path, *, commit_interval: int = 1000) -> None:
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(f"refusing to reuse dedup index {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.executescript(
            """
            CREATE TABLE exact (digest TEXT PRIMARY KEY, document_id TEXT NOT NULL);
            CREATE TABLE signatures (
                document_id TEXT PRIMARY KEY,
                signature TEXT NOT NULL
            );
            CREATE TABLE bands (
                band INTEGER NOT NULL,
                bucket INTEGER NOT NULL,
                document_id TEXT NOT NULL
            );
            CREATE INDEX bands_lookup ON bands(band, bucket);
            """
        )
        self.commit_interval = commit_interval
        self.pending = 0

    def check_and_add(
        self, document: CorpusDocument, *, near_duplicate_hamming: int
    ) -> Rejection | None:
        existing_id = self.connection.execute(
            "SELECT 1 FROM signatures WHERE document_id = ?", (document.id,)
        ).fetchone()
        if existing_id is not None:
            return Rejection(document.id, "duplicate_document_id", document.id)
        digest = document.sha256
        exact = self.connection.execute(
            "SELECT document_id FROM exact WHERE digest = ?", (digest,)
        ).fetchone()
        if exact is not None:
            return Rejection(document.id, "exact_duplicate", str(exact[0]))

        signature = simhash64(document.text)
        clauses = []
        parameters: list[int] = []
        for band in range(4):
            clauses.append("(b.band = ? AND b.bucket = ?)")
            parameters.extend((band, (signature >> (16 * band)) & 0xFFFF))
        candidates = self.connection.execute(
            "SELECT DISTINCT s.document_id, s.signature "
            "FROM bands b JOIN signatures s ON s.document_id = b.document_id "
            f"WHERE {' OR '.join(clauses)} ORDER BY s.document_id",
            parameters,
        ).fetchall()
        duplicate = next(
            (
                str(document_id)
                for document_id, candidate_signature in candidates
                if hamming_distance(signature, int(candidate_signature, 16))
                <= near_duplicate_hamming
            ),
            None,
        )
        if duplicate is not None:
            return Rejection(document.id, "near_duplicate", duplicate)

        self.connection.execute(
            "INSERT INTO exact VALUES (?, ?)", (digest, document.id)
        )
        self.connection.execute(
            "INSERT INTO signatures VALUES (?, ?)",
            (document.id, f"{signature:016x}"),
        )
        self.connection.executemany(
            "INSERT INTO bands VALUES (?, ?, ?)",
            [
                (band, (signature >> (16 * band)) & 0xFFFF, document.id)
                for band in range(4)
            ],
        )
        self.pending += 1
        if self.pending >= self.commit_interval:
            self.connection.commit()
            self.pending = 0
        return None

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


class _RejectionWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.temporary = path.with_suffix(path.suffix + ".tmp")
        raw = self.temporary.open("wb")
        compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
        self.raw = raw
        self.text = io.TextIOWrapper(compressed, encoding="utf-8", newline="\n")

    def write(self, rejection: Rejection) -> None:
        self.text.write(json.dumps(asdict(rejection), sort_keys=True) + "\n")

    def close(self) -> None:
        self.text.flush()
        self.text.close()
        if not self.raw.closed:
            self.raw.close()
        os.replace(self.temporary, self.path)


class StreamingCorpusBuilder:
    """Policy, quality, contamination and dedup gates feeding deterministic shards."""

    def __init__(
        self,
        output_directory: str | Path,
        *,
        registry: SourceRegistry,
        policy: DataPolicy,
        max_shard_bytes: int = 256 * 1024 * 1024,
        min_characters: int = 200,
        max_repeated_line_ratio: float = 0.3,
        reject_pii: bool = True,
        near_duplicate_hamming: int = 3,
        contamination_index: ContaminationIndex | None = None,
        contamination_threshold: float = 0.2,
    ) -> None:
        if min_characters < 1 or near_duplicate_hamming < 0:
            raise ValueError("invalid corpus quality or deduplication setting")
        if not 0 <= max_repeated_line_ratio <= 1:
            raise ValueError("repeated-line threshold must be in [0, 1]")
        if not 0 <= contamination_threshold <= 1:
            raise ValueError("contamination threshold must be in [0, 1]")
        self.output = Path(output_directory)
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output / "manifest.json").exists():
            raise FileExistsError("output directory already contains a corpus manifest")
        self.registry = registry
        self.policy = policy
        self.writer = DeterministicShardWriter(
            self.output / "shards", max_uncompressed_bytes=max_shard_bytes
        )
        self.dedup = SQLiteDedupIndex(self.output / "dedup.sqlite")
        self.rejections = _RejectionWriter(self.output / "rejections.jsonl.gz")
        self.min_characters = min_characters
        self.max_repeated_line_ratio = max_repeated_line_ratio
        self.reject_pii = reject_pii
        self.near_duplicate_hamming = near_duplicate_hamming
        self.contamination_index = contamination_index
        self.contamination_threshold = contamination_threshold
        self.rejection_counts: Counter[str] = Counter()
        self.split_counts: Counter[str] = Counter()
        self.language_counts: Counter[str] = Counter()
        self.domain_counts: Counter[str] = Counter()
        self.license_counts: Counter[str] = Counter()
        self.source_counts: Counter[str] = Counter()
        self.source_status_counts: Counter[str] = Counter()
        self.total_document_bytes = 0
        self.corpus_digest = hashlib.sha256()
        self.finished = False

    def _reject(self, rejection: Rejection) -> None:
        self.rejection_counts[rejection.reason] += 1
        self.rejections.write(rejection)

    def _quality_reason(self, document: CorpusDocument) -> str | None:
        report = assess_quality(document.text)
        if report.characters < self.min_characters:
            return "too_short"
        if report.repeated_line_ratio > self.max_repeated_line_ratio:
            return "repeated_lines"
        if report.replacement_character_ratio > 0.01:
            return "encoding_damage"
        if report.has_secret_pattern:
            return "secret_pattern"
        if self.reject_pii and (report.has_email or report.has_phone):
            return "pii_pattern"
        if self.contamination_index is not None and (
            self.contamination_index.overlap_ratio(document.text)
            >= self.contamination_threshold
        ):
            return "evaluation_contamination"
        return None

    def build(self, documents: Iterable[CorpusDocument]) -> dict[str, Any]:
        if self.finished:
            raise RuntimeError("corpus builder can only run once")
        try:
            for raw in documents:
                normalized = replace(raw, text=normalize_text(raw.text))
                decision = self.registry.decide(normalized, self.policy)
                if not decision.accepted:
                    self._reject(Rejection(normalized.id, decision.reason or "policy"))
                    continue
                document = replace(normalized, license=decision.canonical_license)
                quality_reason = self._quality_reason(document)
                if quality_reason is not None:
                    self._reject(Rejection(document.id, quality_reason))
                    continue
                duplicate = self.dedup.check_and_add(
                    document, near_duplicate_hamming=self.near_duplicate_hamming
                )
                if duplicate is not None:
                    self._reject(duplicate)
                    continue

                split = stable_split(document)
                self.writer.write(split, document)
                self.split_counts[split] += 1
                self.language_counts[document.language] += 1
                self.domain_counts[document.domain] += 1
                self.license_counts[document.license] += 1
                self.source_counts[document.source] += 1
                assert decision.source is not None
                self.source_status_counts[decision.source.status] += 1
                byte_count = len(document.text.encode("utf-8"))
                self.total_document_bytes += byte_count
                self.corpus_digest.update(split.encode())
                self.corpus_digest.update(document.id.encode())
                self.corpus_digest.update(document.sha256.encode())
        finally:
            shards = self.writer.close()
            self.dedup.close()
            self.rejections.close()
            self.finished = True

        manifest: dict[str, Any] = {
            "schema_version": 1,
            "policy": self.policy.to_dict(),
            "source_registry_sha256": self.registry.snapshot_sha256,
            "documents": sum(self.split_counts.values()),
            "utf8_document_bytes": self.total_document_bytes,
            "splits": dict(sorted(self.split_counts.items())),
            "languages": dict(sorted(self.language_counts.items())),
            "domains": dict(sorted(self.domain_counts.items())),
            "licenses": dict(sorted(self.license_counts.items())),
            "sources": dict(sorted(self.source_counts.items())),
            "source_statuses": dict(sorted(self.source_status_counts.items())),
            "rejections": dict(sorted(self.rejection_counts.items())),
            "corpus_sha256": self.corpus_digest.hexdigest(),
            "shards": [asdict(shard) for shard in shards],
            "settings": {
                "min_characters": self.min_characters,
                "max_repeated_line_ratio": self.max_repeated_line_ratio,
                "reject_pii": self.reject_pii,
                "near_duplicate_hamming": self.near_duplicate_hamming,
                "contamination_threshold": self.contamination_threshold,
                "max_shard_bytes": self.writer.max_uncompressed_bytes,
            },
        }
        temporary = self.output / "manifest.json.tmp"
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.output / "manifest.json")
        return manifest


def read_sharded_documents(
    root: str | Path, *, split: str | None = None, verify_hashes: bool = True
) -> Iterator[CorpusDocument]:
    corpus_root = Path(root)
    manifest = json.loads((corpus_root / "manifest.json").read_text(encoding="utf-8"))
    for shard in manifest["shards"]:
        if split is not None and shard["split"] != split:
            continue
        path = corpus_root / "shards" / shard["path"]
        if verify_hashes:
            digest = hashlib.sha256()
            with path.open("rb") as binary:
                for chunk in iter(lambda: binary.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != shard["sha256"]:
                raise ValueError(f"corpus shard hash mismatch: {path}")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                raw = json.loads(line)
                expected_hash = raw.pop("_content_sha256")
                document = CorpusDocument(**raw)
                if verify_hashes and document.sha256 != expected_hash:
                    raise ValueError(f"document hash mismatch: {document.id}")
                yield document
