from __future__ import annotations

import json
from pathlib import Path

import pytest

from minillm.aira.on_policy import (
    OnPolicyTopKRecord,
    collect_on_policy_records,
    read_on_policy_dataset,
    records_to_teacher_topk,
    write_on_policy_dataset,
)


def test_collector_labels_actual_student_generated_prefixes() -> None:
    seen_prefixes: list[bytes] = []

    def student(prompt: bytes, length: int) -> bytes:
        return b"x" * length

    def teacher(prefix: bytes) -> tuple[tuple[int, ...], tuple[float, ...], float]:
        seen_prefixes.append(prefix)
        return (32, 10), (3.0, 1.0), 0.9

    records = collect_on_policy_records(
        [("prompt", b"hello")],
        student_rollout=student,
        teacher_label=teacher,
        rollout_bytes=4,
        label_stride=2,
        teacher_id="teacher-sha",
        student_id="student-sha",
    )

    assert seen_prefixes == [b"hello", b"helloxx", b"helloxxxx"]
    assert [record.identifier for record in records] == [
        "prompt:0",
        "prompt:2",
        "prompt:4",
    ]
    teacher_batch = records_to_teacher_topk(records)
    assert teacher_batch.indices.shape == (3, 2)
    assert teacher_batch.logits.shape == (3, 2)
    assert teacher_batch.mass.shape == (3,)


def test_on_policy_jsonl_round_trip_is_hashed(tmp_path: Path) -> None:
    record = OnPolicyTopKRecord(
        identifier="case:1",
        prefix=b"generated prefix",
        teacher_indices=(1, 2),
        teacher_logits=(4.0, 2.0),
        teacher_mass=0.95,
        teacher_id="teacher",
        student_id="student",
    )
    path = tmp_path / "on-policy.jsonl"

    manifest_path = write_on_policy_dataset(
        path, [record], metadata={"purpose": "test"}
    )
    restored = read_on_policy_dataset(path)
    manifest = json.loads(manifest_path.read_text())

    assert restored == [record]
    assert manifest["records"] == 1
    assert manifest["metadata"] == {"purpose": "test"}
    assert len(manifest["sha256"]) == 64


def test_on_policy_reader_rejects_tampered_prefix_hash(tmp_path: Path) -> None:
    record = OnPolicyTopKRecord(
        identifier="case:1",
        prefix=b"prefix",
        teacher_indices=(1,),
        teacher_logits=(1.0,),
        teacher_mass=0.8,
        teacher_id="teacher",
        student_id="student",
    )
    payload = record.to_dict()
    payload["prefix_sha256"] = "0" * 64
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(payload) + "\n")

    with pytest.raises(ValueError, match="line 1"):
        read_on_policy_dataset(path)
