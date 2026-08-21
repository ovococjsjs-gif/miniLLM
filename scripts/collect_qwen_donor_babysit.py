#!/usr/bin/env python3
"""Convert a Qwen donor rollout into verifier-backed Babysit corrections."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any


def load_lightweight_module(name: str, relative_path: str) -> ModuleType:
    """Load verifier helpers without importing the Torch AIra package."""

    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_verification = load_lightweight_module(
    "minillm_babysit_verification", "src/minillm/aira/verification.py"
)
synthetic_generation_components = _verification.synthetic_generation_components


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def first_difference(first: str, second: str) -> int:
    for index, (left, right) in enumerate(zip(first, second)):
        if left != right:
            return index
    return min(len(first), len(second))


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def prompt_text(record: dict[str, Any]) -> str:
    messages = {message["role"]: message["content"] for message in record["messages"]}
    return f"SYSTEM:\n{messages['system']}\nUSER:\n{messages['user']}"


def critique(language: str, failed: list[str]) -> str:
    joined = ", ".join(failed)
    if language == "ru":
        return (
            f"Ответ не прошёл проверки: {joined}. Привяжи значения к текущему "
            "запросу, соблюдай требуемый протокол и указывай точный источник."
        )
    return (
        f"The answer failed these checks: {joined}. Bind values to the current "
        "request, follow the required protocol, and preserve the exact source."
    )


def protected_content_hashes() -> set[str]:
    """Return immutable Mentor v1 content hashes that must remain evaluation-only."""

    root = Path(__file__).resolve().parents[1] / "artifacts" / "aira-mentor-v1"
    hashes: set[str] = set()
    for filename in ("train.jsonl", "validation.jsonl", "test.jsonl"):
        path = root / filename
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get("content_sha256"):
                    hashes.add(record["content_sha256"])
    return hashes


def collect(
    *,
    tasks_path: Path,
    evaluation_path: Path,
    checkpoint_path: Path,
    task_seed: int,
    output: Path,
    allow_protected: bool = False,
) -> dict[str, Any]:
    """Build exact correction records after independently re-running all verifiers."""

    task_records = [
        json.loads(line)
        for line in tasks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tasks = {record["id"]: record for record in task_records}
    if len(tasks) != len(task_records):
        raise ValueError("task IDs must be unique")
    if not allow_protected:
        overlaps = {
            record.get("content_sha256")
            for record in task_records
            if record.get("content_sha256") in protected_content_hashes()
        }
        if overlaps:
            raise ValueError(
                "tasks overlap protected AIra Mentor v1 content; refusing Babysit export"
            )

    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    samples = evaluation["samples"]
    if len({sample["id"] for sample in samples}) != len(samples):
        raise ValueError("evaluation sample IDs must be unique")
    if {sample["id"] for sample in samples} != set(tasks):
        raise ValueError("evaluation samples and task IDs differ")

    verified_samples: list[tuple[dict[str, Any], dict[str, bool]]] = []
    for sample in samples:
        task = tasks[sample["id"]]
        generated = sample["answer"]
        components = synthetic_generation_components(task, generated)
        if sample.get("components") != components:
            raise ValueError(
                f"component verification mismatch for evaluation sample {sample['id']}"
            )
        if bool(sample.get("passed")) != components["strict"]:
            raise ValueError(f"strict verdict mismatch for evaluation sample {sample['id']}")
        expected_answer_hash = hashlib.sha256(generated.encode()).hexdigest()
        if sample.get("answer_sha256", expected_answer_hash) != expected_answer_hash:
            raise ValueError(f"answer hash mismatch for evaluation sample {sample['id']}")
        verified_samples.append((sample, components))

    checkpoint_hash = sha256(checkpoint_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    verdicts: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    failure_clusters: Counter[str] = Counter()
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for sample, components in verified_samples:
            task = tasks[sample["id"]]
            reference = next(
                message["content"]
                for message in task["messages"]
                if message["role"] == "assistant"
            )
            generated = sample["answer"]
            passed = bool(components["strict"])
            failed = [
                name
                for name in ("content", "protocol", "source")
                if (name != "source" or components["source_required"])
                and not components[name]
            ]
            if not passed and not failed:
                failed = ["strict_surface"]
            if not passed:
                failure_clusters["+".join(failed)] += 1
            verdict = "correct" if passed else "incorrect"
            verdicts[verdict] += 1
            categories[task["category"]] += 1
            correction = generated if passed else reference
            observations = [
                {
                    "tool": "aira-strict-generated-answer",
                    "passed": passed,
                    "detail": json.dumps(
                        {
                            "category": task["category"],
                            "verification": task["verification"],
                            "components": components,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                }
            ]
            payload = {
                "task_id": f"qwen-babysit-seed-{task_seed}:{task['id']}",
                "prompt": prompt_text(task),
                "student_checkpoint": checkpoint_hash,
                "teacher_id": "aira-deterministic-reference-v1",
                "student_answer": generated,
                "verifier_observations": observations,
                "verdict": verdict,
                "critique": "" if passed else critique(task["language"], failed),
                "corrected_answer": correction,
                "rubric_scores": {
                    "strict": float(passed),
                    "correctness": float(components["content"]),
                    "format": float(components["protocol"]),
                    "grounding": float(
                        components["source"]
                        if components["source_required"]
                        else True
                    ),
                },
                "teacher_confidence": 1.0,
                "first_error_offset": (
                    None if passed else first_difference(generated, reference)
                ),
                "error_type": None if passed else "failed_" + "+".join(failed),
                "constitution_flags": [],
                "prompt_sha256": text_hash(prompt_text(task)),
                "student_answer_sha256": text_hash(generated),
                "corrected_answer_sha256": text_hash(correction),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    temporary.replace(output)
    manifest = {
        "schema_version": 1,
        "records": len(samples),
        "sha256": sha256(output),
        "student_checkpoints": [checkpoint_hash],
        "teacher_ids": ["aira-deterministic-reference-v1"],
        "verdicts": dict(sorted(verdicts.items())),
        "metadata": {
            "model_role": "language-donor-and-control-not-teacher",
            "task_seed": task_seed,
            "tasks_sha256": sha256(tasks_path),
            "evaluation_sha256": sha256(evaluation_path),
            "protected_aira_mentor_v1_splits_used": False,
            "categories": dict(sorted(categories.items())),
            "failure_clusters": dict(sorted(failure_clusters.items())),
        },
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = {
        "schema_version": 1,
        "records": len(samples),
        "passed": verdicts["correct"],
        "failed": verdicts["incorrect"],
        "categories": dict(sorted(categories.items())),
        "failure_clusters": dict(sorted(failure_clusters.items())),
        "manifest": str(manifest_path),
    }
    (output.parent / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task-seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allow-protected",
        action="store_true",
        help="unsafe override for verifier tests only; never use for weight-producing exports",
    )
    args = parser.parse_args()
    report = collect(
        tasks_path=Path(args.tasks),
        evaluation_path=Path(args.evaluation),
        checkpoint_path=Path(args.checkpoint),
        task_seed=args.task_seed,
        output=Path(args.output),
        allow_protected=args.allow_protected,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
