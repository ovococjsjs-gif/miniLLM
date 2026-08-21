"""Auditable on-policy teacher-label records for generated student states."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..distillation import TeacherTopK


@dataclass(frozen=True)
class OnPolicyTopKRecord:
    identifier: str
    prefix: bytes
    teacher_indices: tuple[int, ...]
    teacher_logits: tuple[float, ...]
    teacher_mass: float
    teacher_id: str
    student_id: str

    def validate(self, *, vocabulary_size: int = 256) -> None:
        if not self.identifier or not self.prefix:
            raise ValueError(
                "on-policy records need an identifier and non-empty prefix"
            )
        if (
            len(self.teacher_indices) != len(self.teacher_logits)
            or not self.teacher_indices
        ):
            raise ValueError("teacher Top-K indices/logits differ or are empty")
        if len(set(self.teacher_indices)) != len(self.teacher_indices):
            raise ValueError("teacher Top-K indices must be unique")
        if any(not 0 <= index < vocabulary_size for index in self.teacher_indices):
            raise ValueError("teacher index is outside the vocabulary")
        if any(not math.isfinite(value) for value in self.teacher_logits):
            raise ValueError("teacher logits must be finite")
        if not 0 < self.teacher_mass <= 1:
            raise ValueError("teacher retained mass must be in (0, 1]")
        if not self.teacher_id or not self.student_id:
            raise ValueError("teacher and student identities are required")

    @property
    def prefix_sha256(self) -> str:
        return hashlib.sha256(self.prefix).hexdigest()

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "identifier": self.identifier,
            "prefix_base64": base64.b64encode(self.prefix).decode("ascii"),
            "prefix_sha256": self.prefix_sha256,
            "teacher_indices": list(self.teacher_indices),
            "teacher_logits": list(self.teacher_logits),
            "teacher_mass": self.teacher_mass,
            "teacher_id": self.teacher_id,
            "student_id": self.student_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OnPolicyTopKRecord:
        prefix = base64.b64decode(payload["prefix_base64"], validate=True)
        record = cls(
            identifier=str(payload["identifier"]),
            prefix=prefix,
            teacher_indices=tuple(map(int, payload["teacher_indices"])),
            teacher_logits=tuple(map(float, payload["teacher_logits"])),
            teacher_mass=float(payload["teacher_mass"]),
            teacher_id=str(payload["teacher_id"]),
            student_id=str(payload["student_id"]),
        )
        record.validate()
        if record.prefix_sha256 != payload["prefix_sha256"]:
            raise ValueError("on-policy prefix hash mismatch")
        return record


TeacherLabel = Callable[[bytes], tuple[tuple[int, ...], tuple[float, ...], float]]
StudentRollout = Callable[[bytes, int], bytes]


def collect_on_policy_records(
    prompts: Iterable[tuple[str, bytes]],
    *,
    student_rollout: StudentRollout,
    teacher_label: TeacherLabel,
    rollout_bytes: int,
    label_stride: int = 1,
    teacher_id: str,
    student_id: str,
) -> list[OnPolicyTopKRecord]:
    """Ask the teacher to label exact prefixes produced by the student rollout."""

    if rollout_bytes < 1 or label_stride < 1:
        raise ValueError("rollout length and label stride must be positive")
    records = []
    for prompt_id, prompt in prompts:
        if not prompt_id or not prompt:
            raise ValueError("prompts need identifiers and non-empty bytes")
        rollout = student_rollout(prompt, rollout_bytes)
        if len(rollout) != rollout_bytes:
            raise ValueError("student rollout returned the wrong byte count")
        for offset in range(0, rollout_bytes + 1, label_stride):
            prefix = prompt + rollout[:offset]
            indices, logits, mass = teacher_label(prefix)
            record = OnPolicyTopKRecord(
                identifier=f"{prompt_id}:{offset}",
                prefix=prefix,
                teacher_indices=indices,
                teacher_logits=logits,
                teacher_mass=mass,
                teacher_id=teacher_id,
                student_id=student_id,
            )
            record.validate()
            records.append(record)
    return records


def write_on_policy_dataset(
    path: str | Path,
    records: Iterable[OnPolicyTopKRecord],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically write JSONL records and a source/hash manifest."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    count = 0
    teacher_ids: set[str] = set()
    student_ids: set[str] = set()
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
            count += 1
            teacher_ids.add(record.teacher_id)
            student_ids.add(record.student_id)
    if count == 0:
        temporary.unlink(missing_ok=True)
        raise ValueError("cannot write an empty on-policy dataset")
    os.replace(temporary, output)
    with output.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    manifest = {
        "schema_version": 1,
        "records": count,
        "sha256": digest,
        "teacher_ids": sorted(teacher_ids),
        "student_ids": sorted(student_ids),
        "metadata": dict(metadata or {}),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def read_on_policy_dataset(path: str | Path) -> list[OnPolicyTopKRecord]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(OnPolicyTopKRecord.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid on-policy record at line {line_number}"
                ) from error
    if not records:
        raise ValueError("on-policy dataset is empty")
    return records


def records_to_teacher_topk(
    records: list[OnPolicyTopKRecord],
    *,
    device: torch.device | str | None = None,
) -> TeacherTopK:
    """Convert equal-K byte-teacher records to the existing sparse KL format."""

    if not records:
        raise ValueError("teacher record batch cannot be empty")
    width = len(records[0].teacher_indices)
    if any(len(record.teacher_indices) != width for record in records):
        raise ValueError("teacher records in one batch must use equal K")
    return TeacherTopK(
        indices=torch.tensor(
            [record.teacher_indices for record in records],
            dtype=torch.long,
            device=device,
        ),
        logits=torch.tensor(
            [record.teacher_logits for record in records],
            dtype=torch.float32,
            device=device,
        ),
        mass=torch.tensor(
            [record.teacher_mass for record in records],
            dtype=torch.float32,
            device=device,
        ),
    )
