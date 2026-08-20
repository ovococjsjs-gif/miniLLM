from pathlib import Path

from minillm.corpus import ContaminationIndex, CorpusDocument, stable_split
from minillm.corpus_streaming import StreamingCorpusBuilder, read_sharded_documents
from minillm.data_policy import DataPolicy, SourceRegistry
from minillm.importers import common_corpus_documents

ROOT = Path(__file__).parents[1]


def document(
    identifier: str,
    text: str,
    *,
    source: str = "minillm-generated",
    license_name: str = "CC0-1.0",
    language: str = "en",
    domain: str = "dialogue",
    url: str | None = None,
    metadata: dict[str, object] | None = None,
) -> CorpusDocument:
    return CorpusDocument(
        id=identifier,
        text=text,
        source=source,
        license=license_name,
        language=language,
        domain=domain,
        acquisition_date="2026-08-20",
        url=url,
        metadata=metadata or {},
    )


def test_production_policy_rejects_status_license_and_missing_provenance() -> None:
    registry = SourceRegistry.load(ROOT / "configs/corpus/source_registry.json")
    policy = DataPolicy.load(ROOT / "configs/corpus/policy_production.json")
    generated = registry.decide(
        document("ok", "A sufficiently long generated text."), policy
    )
    assert generated.accepted

    openstax = registry.decide(
        document(
            "nc",
            "A restricted textbook sample.",
            source="openstax",
            license_name="CC-BY-NC-SA-4.0",
            domain="education",
        ),
        policy,
    )
    assert not openstax.accepted and openstax.reason == "source_status_research-only"

    missing_url = registry.decide(
        document(
            "by",
            "An attributed source without its URL.",
            source="common-corpus-permissive",
            license_name="cc by 4.0",
            domain="science",
        ),
        policy,
    )
    assert not missing_url.accepted and missing_url.reason == "missing_provenance_url"
    with_url = registry.decide(
        document(
            "by-url",
            "An attributed source with provenance.",
            source="common-corpus-permissive",
            license_name="cc by 4.0",
            domain="science",
            url="https://example.test/document",
        ),
        policy,
    )
    assert with_url.accepted and with_url.canonical_license == "CC-BY-4.0"
    ambiguous_version = registry.decide(
        document(
            "by-ambiguous",
            "A source with an unspecified Creative Commons version.",
            source="common-corpus-permissive",
            license_name="cc-by",
            domain="science",
            url="https://example.test/ambiguous",
        ),
        policy,
    )
    assert not ambiguous_version.accepted
    wikinews = registry.decide(
        document(
            "news",
            "A modern openly attributed news article.",
            source="wikinews-text",
            license_name="cc by 2.5",
            domain="news",
            url="https://ru.wikinews.org/wiki/example",
        ),
        policy,
    )
    assert wikinews.accepted and wikinews.canonical_license == "CC-BY-2.5"


def test_split_group_keeps_chunks_together() -> None:
    first = document("book:chunk-1", "First chunk.", metadata={"split_group": "book"})
    second = document("book:chunk-2", "Second chunk.", metadata={"split_group": "book"})
    assert stable_split(first) == stable_split(second)


def test_common_corpus_chunks_share_split_and_keep_provenance() -> None:
    record = {
        "identifier": "https://example.test/open-paper",
        "text": ("First paragraph has useful scientific prose. " * 20)
        + "\n\n"
        + ("Second paragraph remains in the same source work. " * 20),
        "language": "Russian",
        "open_type": "OpenScience",
        "license": "cc by 4.0",
        "title": "A paper",
        "creator": "An author",
    }
    chunks = tuple(
        common_corpus_documents(
            record, acquisition_date="2026-08-20", maximum_characters=500
        )
    )
    assert len(chunks) > 1
    assert {item.language for item in chunks} == {"ru"}
    assert {item.domain for item in chunks} == {"science"}
    assert {item.url for item in chunks} == {record["identifier"]}
    assert len({stable_split(item) for item in chunks}) == 1


def test_streaming_build_is_deterministic_and_auditable(tmp_path: Path) -> None:
    registry = SourceRegistry.load(ROOT / "configs/corpus/source_registry.json")
    policy = DataPolicy.load(ROOT / "configs/corpus/policy_production.json")
    protected = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"
    english = (
        "A compact local assistant should retrieve evidence, use tools, and answer clearly. "
        "This document has distinct educational content for deterministic corpus testing."
    )
    russian = (
        "Локальный помощник должен проверять факты, пользоваться инструментами и отвечать ясно. "
        "Этот документ содержит отдельный русский текст для проверки корпуса."
    )
    inputs = [
        document("accepted-en", english),
        document("accepted-ru", russian, language="ru"),
        document("duplicate", english),
        document("short", "tiny"),
        document(
            "pii",
            "Contact test.person@example.com for a sufficiently long private record.",
        ),
        document(
            "contaminated",
            protected + " trailing words make this document long enough.",
        ),
    ]

    manifests = []
    roots = []
    for run in ("first", "second"):
        output = tmp_path / run
        builder = StreamingCorpusBuilder(
            output,
            registry=registry,
            policy=policy,
            max_shard_bytes=220,
            min_characters=20,
            near_duplicate_hamming=0,
            contamination_index=ContaminationIndex([protected], width=3),
            contamination_threshold=0.2,
        )
        manifests.append(builder.build(inputs))
        roots.append(output)

    first, second = manifests
    assert first["documents"] == 2
    assert first["languages"] == {"en": 1, "ru": 1}
    assert first["source_statuses"] == {"approved": 2}
    assert first["rejections"] == {
        "evaluation_contamination": 1,
        "exact_duplicate": 1,
        "pii_pattern": 1,
        "too_short": 1,
    }
    assert first["corpus_sha256"] == second["corpus_sha256"]
    assert [item["sha256"] for item in first["shards"]] == [
        item["sha256"] for item in second["shards"]
    ]
    restored = list(read_sharded_documents(roots[0]))
    assert {item.id for item in restored} == {"accepted-en", "accepted-ru"}
    assert all(item.license == "CC0-1.0" for item in restored)
