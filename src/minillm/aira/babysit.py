"""Auditable critique, correction, verifier, and preference records for AI Babysit."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

Verdict = Literal["correct", "incorrect", "unsafe", "unverifiable"]


@dataclass(frozen=True)
class VerifierObservation:
    tool: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class BabysitRecord:
    task_id: str
    prompt: str
    student_checkpoint: str
    teacher_id: str
    student_answer: str
    verifier_observations: tuple[VerifierObservation, ...]
    verdict: Verdict
    critique: str
    corrected_answer: str
    rubric_scores: Mapping[str, float]
    teacher_confidence: float
    first_error_offset: int | None = None
    error_type: str | None = None
    constitution_flags: tuple[str, ...] = ()

    def validate(self) -> None:
        if not all(
            (self.task_id, self.prompt, self.student_checkpoint, self.teacher_id)
        ):
            raise ValueError("babysit identity/prompt fields cannot be empty")
        if self.verdict not in {"correct", "incorrect", "unsafe", "unverifiable"}:
            raise ValueError("invalid babysit verdict")
        if not 0 <= self.teacher_confidence <= 1:
            raise ValueError("teacher confidence must lie in [0, 1]")
        if any(
            not math.isfinite(float(score)) for score in self.rubric_scores.values()
        ):
            raise ValueError("rubric scores must be finite")
        if self.first_error_offset is not None and not (
            0 <= self.first_error_offset <= len(self.student_answer)
        ):
            raise ValueError("first error offset is outside the student answer")
        if self.verdict in {"incorrect", "unsafe"} and (
            not self.critique or not self.corrected_answer or not self.error_type
        ):
            raise ValueError(
                "failed answers require critique, correction, and error type"
            )
        if self.verdict == "correct" and self.corrected_answer not in {
            "",
            self.student_answer,
        }:
            raise ValueError("a correct answer cannot have a different correction")
        if not self.verifier_observations and self.verdict == "correct":
            raise ValueError("correct answers need at least one verifier observation")

    @staticmethod
    def _sha256(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["prompt_sha256"] = self._sha256(self.prompt)
        payload["student_answer_sha256"] = self._sha256(self.student_answer)
        payload["corrected_answer_sha256"] = self._sha256(self.corrected_answer)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BabysitRecord:
        observations = tuple(
            VerifierObservation(**observation)
            for observation in payload["verifier_observations"]
        )
        record = cls(
            task_id=str(payload["task_id"]),
            prompt=str(payload["prompt"]),
            student_checkpoint=str(payload["student_checkpoint"]),
            teacher_id=str(payload["teacher_id"]),
            student_answer=str(payload["student_answer"]),
            verifier_observations=observations,
            verdict=cast(Verdict, str(payload["verdict"])),
            critique=str(payload["critique"]),
            corrected_answer=str(payload["corrected_answer"]),
            rubric_scores={
                str(key): float(value)
                for key, value in payload["rubric_scores"].items()
            },
            teacher_confidence=float(payload["teacher_confidence"]),
            first_error_offset=(
                int(payload["first_error_offset"])
                if payload.get("first_error_offset") is not None
                else None
            ),
            error_type=(
                str(payload["error_type"])
                if payload.get("error_type") is not None
                else None
            ),
            constitution_flags=tuple(map(str, payload.get("constitution_flags", ()))),
        )
        record.validate()
        expected = {
            "prompt_sha256": record._sha256(record.prompt),
            "student_answer_sha256": record._sha256(record.student_answer),
            "corrected_answer_sha256": record._sha256(record.corrected_answer),
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError("babysit text hash mismatch")
        return record

    def preference_pair(self) -> tuple[str, str] | None:
        if self.verdict not in {"incorrect", "unsafe"}:
            return None
        return self.corrected_answer, self.student_answer


def write_babysit_dataset(
    path: str | Path,
    records: Iterable[BabysitRecord],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    count = 0
    checkpoints: set[str] = set()
    teachers: set[str] = set()
    verdicts: dict[str, int] = {}
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            count += 1
            checkpoints.add(record.student_checkpoint)
            teachers.add(record.teacher_id)
            verdicts[record.verdict] = verdicts.get(record.verdict, 0) + 1
    if count == 0:
        temporary.unlink(missing_ok=True)
        raise ValueError("cannot write an empty babysit dataset")
    os.replace(temporary, output)
    with output.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    manifest = {
        "schema_version": 1,
        "records": count,
        "sha256": digest,
        "student_checkpoints": sorted(checkpoints),
        "teacher_ids": sorted(teachers),
        "verdicts": dict(sorted(verdicts.items())),
        "metadata": dict(metadata or {}),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def read_babysit_dataset(path: str | Path) -> list[BabysitRecord]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(BabysitRecord.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid babysit record at line {line_number}"
                ) from error
    if not records:
        raise ValueError("babysit dataset is empty")
    return records
