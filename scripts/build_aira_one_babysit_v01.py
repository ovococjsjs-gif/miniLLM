#!/usr/bin/env python3
"""Compile the first two observed AIra One residual failures into Babysit patches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minillm.aira import (
    BabysitRecord,
    VerifierObservation,
    write_babysit_dataset,
)

CORRECTIONS = {
    "sky": {
        "match": "почему небо кажется голубым",
        "task_id": "aira-one-v01:sky-scattering",
        "critique": (
            "Ответ выдумал неверный механизм со светом Луны и Земли. Нужно назвать "
            "рассеяние солнечного света молекулами воздуха."
        ),
        "correction": (
            "Небо кажется голубым из-за рэлеевского рассеяния: молекулы воздуха "
            "сильнее рассеивают короткие синие волны солнечного света, поэтому со "
            "всех направлений к нам приходит больше синего света."
        ),
        "error_type": "factual-mechanism-hallucination",
    },
    "local_cloud": {
        "match": "сравни локальную и облачную модель",
        "task_id": "aira-one-v01:local-cloud-comparison",
        "critique": (
            "Ответ перепутал локальные и облачные вычисления и не дал по два корректных "
            "преимущества каждого варианта."
        ),
        "correction": (
            "Локальная модель: 1) данные остаются на устройстве; 2) работает без "
            "интернета и без платы за каждый запрос. Облачная модель: 1) обычно "
            "доступно больше вычислительной мощности; 2) обновления и масштабирование "
            "берёт на себя провайдер."
        ),
        "error_type": "instruction-and-concept-binding",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", default="results/aira_one_v01_prepatch_runtime_smoke.json"
    )
    parser.add_argument(
        "--output", default="artifacts/aira-one-babysit-v01/records.jsonl"
    )
    args = parser.parse_args()
    source_path = Path(args.source)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    records = []
    for patch in CORRECTIONS.values():
        sample = next(
            item
            for item in source["samples"]
            if patch["match"] in item["prompt"].casefold()
        )
        records.append(
            BabysitRecord(
                task_id=patch["task_id"],
                prompt=sample["prompt"],
                student_checkpoint=source["donor_sha256"],
                teacher_id="arena-agent-teacher-v1",
                student_answer=sample["answer"],
                verifier_observations=(
                    VerifierObservation(
                        tool="aira-one-teacher-review-v1",
                        passed=False,
                        detail=patch["critique"],
                    ),
                ),
                verdict="incorrect",
                critique=patch["critique"],
                corrected_answer=patch["correction"],
                rubric_scores={
                    "correctness": 0.0,
                    "instruction_following": 0.25,
                    "grounding": 0.0,
                    "format": 1.0,
                },
                teacher_confidence=1.0,
                first_error_offset=0,
                error_type=patch["error_type"],
            )
        )
    output = Path(args.output)
    manifest = write_babysit_dataset(
        output,
        records,
        metadata={
            "source": str(source_path),
            "model": "aira-one-v0.1-pre-patch",
            "on_policy": True,
            "hidden_reasoning_stored": False,
            "protected_evaluation_used": False,
        },
    )
    report = {
        "schema_version": 1,
        "records": len(records),
        "failures": {record.error_type: 1 for record in records},
        "manifest": str(manifest),
        "installed_routes": [
            "babysit.skill.sky-scattering-v1",
            "babysit.skill.local-cloud-v1",
        ],
    }
    (output.parent / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output.parent / "README.md").write_text(
        "# AIra One Babysit v0.1\n\n"
        "Two exact on-policy residual failures from the first mixed runtime smoke. "
        "Teacher corrections are installed as high-confidence AIra skill routes; the "
        "donor remains the student, not the teacher.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
