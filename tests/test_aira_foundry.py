from __future__ import annotations

import json
from pathlib import Path

from minillm.aira import BabysitRecord, VerifierObservation
from minillm.aira.foundry import (
    build_teacher_packet,
    compile_curriculum,
    fingerprint_babysit_record,
    mentor_skill_patches_v1,
    write_curriculum_dataset,
)
from minillm.aira.synthetic import generate_aira_mentor_records
from minillm.aira.verification import verify_synthetic_generation


def failed_arithmetic() -> BabysitRecord:
    return BabysitRecord(
        task_id="babysit:aira-mentor-v1-arithmetic-00000",
        prompt=(
            "SYSTEM:\nBe exact.\nUSER:\n"
            "3 machines make 12 parts each for 4 hours; reject 5."
        ),
        student_checkpoint="student-sha",
        teacher_id="deterministic-reference",
        student_answer="6 × 20 × 4 = 480; 480 - 5 = 475. Answer: 475.",
        verifier_observations=(
            VerifierObservation("aira-integer", False, "expected 139"),
        ),
        verdict="incorrect",
        critique="Operands came from another task.",
        corrected_answer="3 × 12 × 4 = 144; 144 - 5 = 139. Answer: 139.",
        rubric_scores={"correctness": 0.0},
        teacher_confidence=1.0,
        first_error_offset=0,
        error_type="failed_integer",
    )


def test_failure_packet_clusters_by_actionable_signature() -> None:
    record = failed_arithmetic()
    fingerprint = fingerprint_babysit_record(record)
    packet = build_teacher_packet([record], packet_id="packet-test")

    assert fingerprint.category == "arithmetic"
    assert fingerprint.failure_mode == "operand-binding-drift"
    assert packet.failed_records == 1
    assert packet.clusters[0].failure_mode == "operand-binding-drift"
    assert packet.to_dict()["content_sha256"] == packet.content_sha256


def test_skill_patches_compile_generated_and_on_policy_curriculum(
    tmp_path: Path,
) -> None:
    packet = build_teacher_packet([failed_arithmetic()], packet_id="packet-test")
    patches = mentor_skill_patches_v1(packet.content_sha256)
    curriculum = compile_curriculum(
        patches,
        generated_examples_per_category=1,
        generated_seed=91,
        babysit_records=[failed_arithmetic()],
    )

    assert len(patches) == 11
    assert len(curriculum) == 11
    assert {record.source_kind for record in curriculum} == {
        "deterministic-generated",
        "on-policy-correction",
    }
    assert all(record.chosen != record.rejected for record in curriculum)
    assert len({record.content_sha256 for record in curriculum}) == len(curriculum)

    manifest_path = write_curriculum_dataset(tmp_path / "curriculum.jsonl", curriculum)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["records"] == 11
    assert manifest["sources"]["on-policy-correction"] == 1


def test_python_verifier_runs_restricted_unit_tests() -> None:
    records = generate_aira_mentor_records(examples_per_category=1, seed=92)
    python_record = next(record for record in records if record.category == "python")
    reference = python_record.messages[-1].content
    function_name = python_record.verification["function"]
    wrong = f"```python\ndef {function_name}(values):\n    return 123\n```"
    unsafe = (
        f"```python\ndef {function_name}(values):\n"
        "    import os\n    return os.listdir('.')\n```"
    )

    assert verify_synthetic_generation(python_record, reference)
    assert not verify_synthetic_generation(python_record, wrong)
    assert not verify_synthetic_generation(python_record, unsafe)
