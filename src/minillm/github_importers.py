"""Pinned GitHub corpus adapters used by the first real-data pilot."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import quote

from .corpus import CorpusDocument
from .importers import chunk_paragraphs


def stable_bucket(identifier: str, modulus: int = 1_000_000) -> int:
    digest = hashlib.blake2s(identifier.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % modulus


def _github_url(repository: str, revision: str, relative_path: Path) -> str:
    base = repository.removesuffix(".git")
    encoded = quote(relative_path.as_posix(), safe="/")
    return f"{base}/blob/{revision}/{encoded}"


def _documents_from_text(
    *,
    identifier_prefix: str,
    text: str,
    source: str,
    license_name: str,
    language: str,
    domain: str,
    acquisition_date: str,
    url: str,
    metadata: dict[str, object],
    maximum_characters: int,
) -> Iterator[CorpusDocument]:
    split_group = identifier_prefix
    for index, chunk in enumerate(chunk_paragraphs(text, maximum_characters)):
        yield CorpusDocument(
            id=f"{identifier_prefix}:chunk-{index:05d}",
            text=chunk,
            source=source,
            license=license_name,
            language=language,
            domain=domain,
            acquisition_date=acquisition_date,
            url=url,
            metadata={**metadata, "split_group": split_group},
        )


def import_oanc_checkout(
    checkout: str | Path,
    *,
    repository: str,
    revision: str,
    acquisition_date: str,
    sample_parts_per_million: int,
    maximum_characters: int,
) -> Iterator[CorpusDocument]:
    root = Path(checkout)
    for path in sorted((root / "oanc").glob("*.txt")):
        relative = path.relative_to(root)
        if stable_bucket(relative.as_posix()) >= sample_parts_per_million:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        prefix = f"oanc:{relative.stem}"
        yield from _documents_from_text(
            identifier_prefix=prefix,
            text=text,
            source="oanc-github-mirror",
            license_name="OANC-Unrestricted",
            language="en",
            domain="general",
            acquisition_date=acquisition_date,
            url=_github_url(repository, revision, relative),
            metadata={
                "repository": repository,
                "repository_revision": revision,
                "repository_path": relative.as_posix(),
            },
            maximum_characters=maximum_characters,
        )


def import_ruslit_checkout(
    checkout: str | Path,
    *,
    repository: str,
    revision: str,
    acquisition_date: str,
    allowed_authors: frozenset[str],
    sample_parts_per_million: int,
    maximum_characters: int,
) -> Iterator[CorpusDocument]:
    root = Path(checkout)
    for path in sorted(root.glob("*/*/*.txt")):
        relative = path.relative_to(root)
        if len(relative.parts) != 3 or relative.parts[1] not in allowed_authors:
            continue
        if stable_bucket(relative.as_posix()) >= sample_parts_per_million:
            continue
        genre, author = relative.parts[:2]
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        digest = hashlib.blake2s(
            relative.as_posix().encode(), digest_size=8
        ).hexdigest()
        prefix = f"ruslit:{digest}"
        yield from _documents_from_text(
            identifier_prefix=prefix,
            text=text,
            source="ruslit-pd-github",
            license_name="Public-Domain",
            language="ru",
            domain="literature",
            acquisition_date=acquisition_date,
            url=_github_url(repository, revision, relative),
            metadata={
                "repository": repository,
                "repository_revision": revision,
                "repository_path": relative.as_posix(),
                "title": path.stem,
                "creator": author,
                "genre": genre,
            },
            maximum_characters=maximum_characters,
        )


def _local_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def extract_tei_text(path: str | Path) -> tuple[str, dict[str, str | None]]:
    root = ET.parse(path).getroot()
    title = author = date = None
    for element in root.iter():
        name = _local_name(element)
        value = " ".join("".join(element.itertext()).split())
        if name == "title" and title is None and value:
            title = value
        elif name == "author" and author is None and value:
            author = value
        elif name in {"date", "origDate"} and date is None and value:
            date = value
    text_root = next(
        (item for item in root.iter() if _local_name(item) == "text"), None
    )
    if text_root is None:
        return "", {"title": title, "creator": author, "source_date": date}
    segment_names = {"head", "speaker", "stage", "p", "l"}
    segments: list[str] = []
    for element in text_root.iter():
        if _local_name(element) not in segment_names:
            continue
        value = " ".join("".join(element.itertext()).split())
        if value:
            segments.append(value)
    if not segments:
        segments.append(" ".join("".join(text_root.itertext()).split()))
    return "\n".join(segments), {
        "title": title,
        "creator": author,
        "source_date": date,
    }


def import_rusdracor_checkout(
    checkout: str | Path,
    *,
    repository: str,
    revision: str,
    acquisition_date: str,
    sample_parts_per_million: int,
    maximum_characters: int,
) -> Iterator[CorpusDocument]:
    root = Path(checkout)
    for path in sorted((root / "tei").glob("*.xml")):
        relative = path.relative_to(root)
        if stable_bucket(relative.as_posix()) >= sample_parts_per_million:
            continue
        text, tei_metadata = extract_tei_text(path)
        prefix = f"rusdracor:{path.stem}"
        yield from _documents_from_text(
            identifier_prefix=prefix,
            text=text,
            source="rusdracor-github",
            license_name="CC0-1.0",
            language="ru",
            domain="drama",
            acquisition_date=acquisition_date,
            url=_github_url(repository, revision, relative),
            metadata={
                "repository": repository,
                "repository_revision": revision,
                "repository_path": relative.as_posix(),
                **tei_metadata,
            },
            maximum_characters=maximum_characters,
        )
