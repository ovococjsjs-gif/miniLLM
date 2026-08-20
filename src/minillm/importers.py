"""Adapters from external open-corpus records into auditable documents."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from .corpus import CorpusDocument

_LANGUAGE_ALIASES = {
    "english": "en",
    "en": "en",
    "russian": "ru",
    "ru": "ru",
}
_DOMAIN_MAP = {
    "openculture": "culture",
    "open culture": "culture",
    "opengovernment": "government",
    "open government": "government",
    "opensource": "code",
    "open source": "code",
    "openscience": "science",
    "open science": "science",
    "openweb": "web",
    "open web": "web",
    "opensemantic": "semantic",
    "open semantic": "semantic",
}


def canonical_common_corpus_language(value: object) -> str | None:
    return _LANGUAGE_ALIASES.get(str(value).casefold())


def chunk_paragraphs(text: str, maximum_characters: int) -> Iterator[str]:
    """Split oversized records at paragraphs while preserving all non-empty text."""

    if maximum_characters < 1:
        raise ValueError("maximum chunk size must be positive")
    current: list[str] = []
    length = 0
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > maximum_characters:
            if current:
                yield "\n\n".join(current)
                current, length = [], 0
            for start in range(0, len(paragraph), maximum_characters):
                yield paragraph[start : start + maximum_characters]
            continue
        added = len(paragraph) + (2 if current else 0)
        if current and length + added > maximum_characters:
            yield "\n\n".join(current)
            current, length = [], 0
        current.append(paragraph)
        length += added
    if current:
        yield "\n\n".join(current)


def common_corpus_documents(
    record: Mapping[str, Any], *, acquisition_date: str, maximum_characters: int
) -> Iterator[CorpusDocument]:
    identifier = str(record.get("identifier", "")).strip()
    text = str(record.get("text", "")).strip()
    language = canonical_common_corpus_language(record.get("language", ""))
    open_type = str(record.get("open_type", record.get("open type", "")))
    domain = _DOMAIN_MAP.get(open_type.casefold())
    if not identifier or not text or language is None or domain is None:
        return
    url_value = record.get("url")
    url = str(url_value).strip() if url_value else None
    if url is None and identifier.startswith(("http://", "https://")):
        url = identifier
    metadata = {
        "split_group": f"common-corpus:{identifier}",
        "collection": record.get("collection"),
        "open_type": open_type,
        "curator": record.get("curator"),
        "title": record.get("title"),
        "creator": record.get("creator"),
        "source_date": record.get("date"),
    }
    for index, chunk in enumerate(chunk_paragraphs(text, maximum_characters)):
        yield CorpusDocument(
            id=f"common-corpus:{identifier}:chunk-{index:05d}",
            text=chunk,
            source="common-corpus-permissive",
            license=str(record.get("license", "")),
            language=language,
            domain=domain,
            acquisition_date=acquisition_date,
            url=url,
            metadata=metadata,
        )
