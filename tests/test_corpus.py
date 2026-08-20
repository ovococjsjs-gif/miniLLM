from minillm.corpus import (
    ContaminationIndex,
    CorpusBuilder,
    CorpusDocument,
    content_hash,
    stable_split,
)


def document(identifier: str, text: str, *, license: str = "CC0") -> CorpusDocument:
    return CorpusDocument(identifier, text, "test", license, "en", "test", "2026-08-20")


def test_exact_near_dedup_and_license_filtering() -> None:
    base = "Neural models can use deterministic tools for exact arithmetic. " * 10
    documents = [
        document("a", base),
        document("b", base),
        document("c", base + "x"),
        document("d", "unlicensed text " * 30, license="proprietary"),
    ]
    result = CorpusBuilder(
        allowed_licenses=frozenset({"CC0"}),
        min_characters=100,
        near_duplicate_hamming=8,
    ).build(documents)
    assert [item.id for item in result.accepted] == ["a"]
    reasons = {item.document_id: item.reason for item in result.rejected}
    assert reasons == {
        "b": "exact_duplicate",
        "c": "near_duplicate",
        "d": "license_not_allowed",
    }


def test_contamination_and_pii_are_rejected() -> None:
    protected = (
        "the quick brown fox jumps over the lazy dog and solves a hidden benchmark task"
    )
    index = ContaminationIndex([protected], width=5)
    contaminated = document("x", (protected + " extra words ") * 8)
    pii = document("y", ("Contact person@example.org for private details. " * 10))
    result = CorpusBuilder(
        min_characters=100,
        contamination_index=index,
        contamination_threshold=0.1,
    ).build([contaminated, pii])
    assert {item.reason for item in result.rejected} == {
        "evaluation_contamination",
        "pii_pattern",
    }


def test_hash_and_split_are_stable() -> None:
    item = document("stable-id", "Text with   spaces.\n")
    assert content_hash(item.text) == content_hash("Text with spaces.")
    assert stable_split(item) == stable_split(item)
