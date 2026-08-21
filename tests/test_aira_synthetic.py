from __future__ import annotations

from collections import Counter

from minillm.aira.synthetic import (
    generate_aira_mentor_records,
    validate_synthetic_record,
)


def test_aira_mentor_generator_is_deterministic_balanced_and_verified() -> None:
    first = generate_aira_mentor_records(examples_per_category=6, seed=42)
    second = generate_aira_mentor_records(examples_per_category=6, seed=42)

    assert [record.to_dict() for record in first] == [
        record.to_dict() for record in second
    ]
    assert len(first) == 60
    assert set(Counter(record.category for record in first).values()) == {6}
    assert Counter(record.language for record in first) == {"ru": 30, "en": 30}
    assert len({record.content_sha256 for record in first}) == len(first)
    assert all(record.verification["verified"] for record in first)


def test_generator_replaces_excluded_content_without_changing_inventory() -> None:
    baseline = generate_aira_mentor_records(examples_per_category=1, seed=47)
    blocked = {record.content_sha256 for record in baseline}

    replacement = generate_aira_mentor_records(
        examples_per_category=1,
        seed=47,
        excluded_content_hashes=blocked,
    )

    assert len(replacement) == len(baseline) == 10
    assert not blocked & {record.content_sha256 for record in replacement}
    assert [record.identifier for record in replacement] == [
        record.identifier for record in baseline
    ]
    assert all(record.provenance["dedup_attempt"] >= 1 for record in replacement)


def test_synthetic_records_have_chat_order_and_provenance() -> None:
    records = generate_aira_mentor_records(examples_per_category=2, seed=7)
    for record in records:
        validate_synthetic_record(record)
        assert [message.role for message in record.messages] == [
            "system",
            "user",
            "assistant",
        ]
        assert record.provenance["teacher"] == "deterministic-project-owned"
        assert record.split in {"train", "validation", "test"}
