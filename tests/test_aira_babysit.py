from __future__ import annotations

import json
from pathlib import Path

import pytest

from minillm.aira.babysit import (
    BabysitRecord,
    VerifierObservation,
    read_babysit_dataset,
    write_babysit_dataset,
)


def incorrect_record() -> BabysitRecord:
    return BabysitRecord(
        task_id="math-1",
        prompt="What is 2 + 2?",
        student_checkpoint="student-sha",
        teacher_id="teacher-sha",
        student_answer="5",
        verifier_observations=(
            VerifierObservation("calculator", False, "expected 4, received 5"),
        ),
        verdict="incorrect",
        critique="The arithmetic result is wrong.",
        corrected_answer="4",
        rubric_scores={"correctness": 0, "style": 1},
        teacher_confidence=1.0,
        first_error_offset=0,
        error_type="arithmetic",
    )


def test_babysit_record_produces_preference_pair() -> None:
    record = incorrect_record()
    record.validate()

    assert record.preference_pair() == ("4", "5")
    assert (
        record.to_dict()["student_answer_sha256"]
        != record.to_dict()["corrected_answer_sha256"]
    )


def test_babysit_dataset_round_trip_and_manifest(tmp_path: Path) -> None:
    path = tmp_path / "babysit.jsonl"
    manifest_path = write_babysit_dataset(
        path, [incorrect_record()], metadata={"constitution": "v1"}
    )

    restored = read_babysit_dataset(path)
    manifest = json.loads(manifest_path.read_text())

    assert restored == [incorrect_record()]
    assert manifest["verdicts"] == {"incorrect": 1}
    assert manifest["metadata"] == {"constitution": "v1"}


def test_failed_answer_requires_actionable_correction() -> None:
    record = incorrect_record()
    invalid = BabysitRecord(
        **{
            **record.__dict__,
            "critique": "",
            "corrected_answer": "",
        }
    )
    with pytest.raises(ValueError, match="require critique"):
        invalid.validate()
