"""Streaming-friendly corpus records, filtering, deduplication, and manifests."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\s().-]?){9,15}(?!\d)")
_SECRET = re.compile(
    r"(?:api[_-]?key|secret|password|token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-/.]{12,}",
    re.IGNORECASE,
)
_WORD = re.compile(r"\w+", re.UNICODE)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\x00", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CorpusDocument:
    id: str
    text: str
    source: str
    license: str
    language: str
    domain: str
    acquisition_date: str
    created_at: str | None = None
    url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> CorpusDocument:
        return CorpusDocument(**{**asdict(self), "text": normalize_text(self.text)})

    @property
    def sha256(self) -> str:
        return content_hash(self.text)


@dataclass(frozen=True)
class QualityReport:
    characters: int
    words: int
    alphabetic_ratio: float
    repeated_line_ratio: float
    replacement_character_ratio: float
    has_email: bool
    has_phone: bool
    has_secret_pattern: bool


@dataclass(frozen=True)
class Rejection:
    document_id: str
    reason: str
    duplicate_of: str | None = None


@dataclass(frozen=True)
class CorpusBuildResult:
    accepted: tuple[CorpusDocument, ...]
    rejected: tuple[Rejection, ...]


def assess_quality(text: str) -> QualityReport:
    characters = len(text)
    words = len(_WORD.findall(text))
    alphabetic = sum(character.isalpha() for character in text)
    lines = [line for line in text.splitlines() if line.strip()]
    counts = Counter(lines)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    replacement = text.count("�")
    return QualityReport(
        characters=characters,
        words=words,
        alphabetic_ratio=alphabetic / max(1, characters),
        repeated_line_ratio=repeated / max(1, len(lines)),
        replacement_character_ratio=replacement / max(1, characters),
        has_email=bool(_EMAIL.search(text)),
        has_phone=bool(_PHONE.search(text)),
        has_secret_pattern=bool(_SECRET.search(text)),
    )


def _shingles(text: str, width: int = 3) -> list[str]:
    words = [word.casefold() for word in _WORD.findall(text)]
    if len(words) < width:
        return [" ".join(words)] if words else []
    return [
        " ".join(words[index : index + width])
        for index in range(len(words) - width + 1)
    ]


def simhash64(text: str) -> int:
    vector = [0] * 64
    for shingle in set(_shingles(text)):
        digest = int.from_bytes(
            hashlib.blake2b(shingle.encode(), digest_size=8).digest(), "big"
        )
        for bit in range(64):
            vector[bit] += 1 if digest & (1 << bit) else -1
    result = 0
    for bit, value in enumerate(vector):
        if value >= 0:
            result |= 1 << bit
    return result


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


class ContaminationIndex:
    """Exact word-shingle index for small protected evaluation sets."""

    def __init__(self, protected_texts: Iterable[str], *, width: int = 13) -> None:
        self.width = width
        self.shingles = {
            hashlib.blake2s(item.encode(), digest_size=8).digest()
            for text in protected_texts
            for item in _shingles(text, width)
        }

    def overlap_ratio(self, text: str) -> float:
        candidates = _shingles(text, self.width)
        if not candidates:
            return 0.0
        overlap = sum(
            hashlib.blake2s(item.encode(), digest_size=8).digest() in self.shingles
            for item in candidates
        )
        return overlap / len(candidates)


class CorpusBuilder:
    """Deterministic quality gate with exact and LSH-assisted near deduplication."""

    def __init__(
        self,
        *,
        allowed_licenses: frozenset[str] | None = None,
        min_characters: int = 200,
        max_repeated_line_ratio: float = 0.3,
        reject_pii: bool = True,
        near_duplicate_hamming: int = 3,
        contamination_index: ContaminationIndex | None = None,
        contamination_threshold: float = 0.2,
    ) -> None:
        self.allowed_licenses = allowed_licenses
        self.min_characters = min_characters
        self.max_repeated_line_ratio = max_repeated_line_ratio
        self.reject_pii = reject_pii
        self.near_duplicate_hamming = near_duplicate_hamming
        self.contamination_index = contamination_index
        self.contamination_threshold = contamination_threshold

    def build(self, documents: Iterable[CorpusDocument]) -> CorpusBuildResult:
        accepted: list[CorpusDocument] = []
        rejected: list[Rejection] = []
        exact: dict[str, str] = {}
        signatures: dict[str, int] = {}
        bands: dict[tuple[int, int], list[str]] = defaultdict(list)
        for raw in documents:
            document = raw.normalized()
            report = assess_quality(document.text)
            reason: str | None = None
            if (
                self.allowed_licenses is not None
                and document.license not in self.allowed_licenses
            ):
                reason = "license_not_allowed"
            elif report.characters < self.min_characters:
                reason = "too_short"
            elif report.repeated_line_ratio > self.max_repeated_line_ratio:
                reason = "repeated_lines"
            elif report.replacement_character_ratio > 0.01:
                reason = "encoding_damage"
            elif report.has_secret_pattern:
                reason = "secret_pattern"
            elif self.reject_pii and (report.has_email or report.has_phone):
                reason = "pii_pattern"
            elif self.contamination_index is not None and (
                self.contamination_index.overlap_ratio(document.text)
                >= self.contamination_threshold
            ):
                reason = "evaluation_contamination"
            if reason:
                rejected.append(Rejection(document.id, reason))
                continue

            digest = document.sha256
            if digest in exact:
                rejected.append(
                    Rejection(document.id, "exact_duplicate", exact[digest])
                )
                continue
            signature = simhash64(document.text)
            candidate_ids: set[str] = set()
            for band in range(4):
                candidate_ids.update(bands[(band, (signature >> (16 * band)) & 0xFFFF)])
            duplicate = next(
                (
                    candidate_id
                    for candidate_id in sorted(candidate_ids)
                    if hamming_distance(signature, signatures[candidate_id])
                    <= self.near_duplicate_hamming
                ),
                None,
            )
            if duplicate is not None:
                rejected.append(Rejection(document.id, "near_duplicate", duplicate))
                continue
            accepted.append(document)
            exact[digest] = document.id
            signatures[document.id] = signature
            for band in range(4):
                bands[(band, (signature >> (16 * band)) & 0xFFFF)].append(document.id)
        return CorpusBuildResult(tuple(accepted), tuple(rejected))


def stable_split(
    document: CorpusDocument, *, train: int = 90, validation: int = 5
) -> str:
    if train < 1 or validation < 1 or train + validation >= 100:
        raise ValueError("invalid split percentages")
    split_group = str(document.metadata.get("split_group", document.id))
    bucket = (
        int(hashlib.blake2s(split_group.encode(), digest_size=4).hexdigest(), 16) % 100
    )
    if bucket < train:
        return "train"
    if bucket < train + validation:
        return "validation"
    return "test"


def write_jsonl(path: str | Path, documents: Iterable[CorpusDocument]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(
                json.dumps(asdict(document), ensure_ascii=False, sort_keys=True) + "\n"
            )


def read_jsonl(path: str | Path) -> Iterator[CorpusDocument]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield CorpusDocument(**json.loads(line))


def build_manifest(documents: Sequence[CorpusDocument]) -> dict[str, Any]:
    by_language = Counter(document.language for document in documents)
    by_domain = Counter(document.domain for document in documents)
    by_license = Counter(document.license for document in documents)
    total_bytes = sum(len(document.text.encode("utf-8")) for document in documents)
    digest = hashlib.sha256()
    for document in sorted(documents, key=lambda item: item.id):
        digest.update(document.id.encode())
        digest.update(document.sha256.encode())
    entropy = 0.0
    for count in by_language.values():
        probability = count / max(1, len(documents))
        entropy -= probability * math.log2(probability)
    return {
        "documents": len(documents),
        "utf8_bytes": total_bytes,
        "by_language": dict(sorted(by_language.items())),
        "by_domain": dict(sorted(by_domain.items())),
        "by_license": dict(sorted(by_license.items())),
        "language_entropy_bits": entropy,
        "manifest_sha256": digest.hexdigest(),
    }
