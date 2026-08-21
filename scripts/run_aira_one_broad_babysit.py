#!/usr/bin/env python3
"""Run three broad AIra One on-policy Babysit cycles and install reviewed skills."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from minillm.aira import (
    AIraMode,
    AIraOne,
    BabysitRecord,
    LocalDonorRuntime,
    OpenAIChatProvider,
    ShelfSkill,
    SkillShelf,
    VerifierObservation,
    write_babysit_dataset,
)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_answer(task: dict[str, Any], answer: str) -> dict[str, Any]:
    folded = answer.casefold()
    missing = [
        group
        for group in task["answer_requirements"]
        if not any(str(term).casefold() in folded for term in group)
    ]
    forbidden = [
        term for term in task.get("forbidden", []) if term.casefold() in folded
    ]
    return {
        "passed": not missing and not forbidden,
        "missing_groups": missing,
        "forbidden_matches": forbidden,
        "nonempty": bool(answer.strip()),
    }


def shelf_skill(task: dict[str, Any], cycle_id: str) -> ShelfSkill:
    required = task["required_groups"]
    return ShelfSkill(
        skill_id=f"{cycle_id}.{task['id']}",
        required_groups_ru=tuple(tuple(group) for group in required["ru"]),
        required_groups_en=tuple(tuple(group) for group in required["en"]),
        answer_ru=task["answers"]["ru"],
        answer_en=task["answers"]["en"],
        provenance="aira-one-broad-babysit-v1",
    )


def skill_payload(skill: ShelfSkill) -> dict[str, Any]:
    return {
        "skill_id": skill.skill_id,
        "required_groups": {
            "ru": [list(group) for group in skill.required_groups_ru],
            "en": [list(group) for group in skill.required_groups_en],
        },
        "answers": {"ru": skill.answer_ru, "en": skill.answer_en},
        "provenance": skill.provenance,
    }


def attempt(
    assistant: AIraOne,
    prompt: str,
    task: dict[str, Any],
) -> dict[str, Any]:
    response = assistant.answer(prompt, mode=AIraMode.BALANCED)
    verification = verify_answer(task, response.answer)
    return {
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "answer": response.answer,
        "answer_sha256": hashlib.sha256(response.answer.encode()).hexdigest(),
        "route": response.route,
        "model_bypassed": response.model_bypassed,
        "neural_calls": response.neural_calls,
        "latency_seconds": response.latency_seconds,
        "verification": verification,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--curriculum", default="configs/aira-one/broad_curriculum_v1.json"
    )
    parser.add_argument(
        "--model",
        default=(
            "data/external/qwen3.5-0.8b/"
            "Qwen3.5-0.8B-Q4_K_M-github.gguf"
        ),
    )
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument(
        "--artifact-dir", default="artifacts/aira-one-broad-babysit-v1"
    )
    parser.add_argument(
        "--output", default="results/aira_one_broad_babysit_v1.json"
    )
    args = parser.parse_args()

    curriculum_path = Path(args.curriculum)
    curriculum = json.loads(curriculum_path.read_text(encoding="utf-8"))
    tasks = [task for cycle in curriculum["cycles"] for task in cycle["tasks"]]
    for task in tasks:
        gold = task["answers"][task["language"]]
        check = verify_answer(task, gold)
        if not check["passed"]:
            raise ValueError(f"teacher answer fails rubric for {task['id']}: {check}")
        probe_skill = shelf_skill(task, "probe")
        if SkillShelf((probe_skill,)).match(
            task["validation_prompt"], ru=task["language"] == "ru"
        ) is None:
            raise ValueError(f"validation paraphrase does not match skill {task['id']}")

    model_hash = sha256(Path(args.model))
    runtime = LocalDonorRuntime(
        model=args.model,
        port=args.port,
        log_path=".aira-one/broad-babysit-llama.log",
    )
    runtime.start()
    provider = OpenAIChatProvider(
        base_url=runtime.endpoint,
        model="aira-one-donor",
        timeout_seconds=240,
    )
    installed: list[ShelfSkill] = []
    babysit_records = []
    cycle_reports = []
    started = time.perf_counter()
    try:
        for cycle in curriculum["cycles"]:
            baseline = AIraOne(provider, skill_shelf=SkillShelf(tuple(installed)))
            train_attempts = []
            validation_before = []
            failed_tasks = []
            for task in cycle["tasks"]:
                train = attempt(baseline, task["train_prompt"], task)
                train_attempts.append({"task_id": task["id"], **train})
                before = attempt(baseline, task["validation_prompt"], task)
                validation_before.append({"task_id": task["id"], **before})
                passed = bool(train["verification"]["passed"])
                teacher_answer = task["answers"][task["language"]]
                detail = json.dumps(train["verification"], ensure_ascii=False, sort_keys=True)
                babysit_records.append(
                    BabysitRecord(
                        task_id=f"aira-one-broad:{cycle['id']}:{task['id']}",
                        prompt=task["train_prompt"],
                        student_checkpoint=model_hash,
                        teacher_id="arena-agent-teacher-v1",
                        student_answer=train["answer"],
                        verifier_observations=(
                            VerifierObservation(
                                tool="aira-broad-concept-rubric-v1",
                                passed=passed,
                                detail=detail,
                            ),
                        ),
                        verdict="correct" if passed else "incorrect",
                        critique=(
                            ""
                            if passed
                            else "Ответ не покрывает обязательные проверяемые понятия или содержит запрещённое утверждение."
                        ),
                        corrected_answer=train["answer"] if passed else teacher_answer,
                        rubric_scores={
                            "correctness": float(passed),
                            "concept_coverage": float(passed),
                            "format": float(bool(train["answer"].strip())),
                        },
                        teacher_confidence=1.0,
                        first_error_offset=None if passed else 0,
                        error_type=None if passed else "broad-concept-coverage",
                    )
                )
                if not passed:
                    failed_tasks.append(task)
                    installed.append(shelf_skill(task, cycle["id"]))

            patched = AIraOne(provider, skill_shelf=SkillShelf(tuple(installed)))
            validation_after = [
                {
                    "task_id": task["id"],
                    **attempt(patched, task["validation_prompt"], task),
                }
                for task in cycle["tasks"]
            ]
            cycle_reports.append(
                {
                    "cycle": cycle["id"],
                    "tasks": len(cycle["tasks"]),
                    "train_passes": sum(
                        item["verification"]["passed"] for item in train_attempts
                    ),
                    "validation_passes_before": sum(
                        item["verification"]["passed"] for item in validation_before
                    ),
                    "validation_passes_after": sum(
                        item["verification"]["passed"] for item in validation_after
                    ),
                    "skills_installed": len(failed_tasks),
                    "neural_calls_before": sum(
                        item["neural_calls"]
                        for item in [*train_attempts, *validation_before]
                    ),
                    "validation_neural_calls_after": sum(
                        item["neural_calls"] for item in validation_after
                    ),
                    "train_attempts": train_attempts,
                    "validation_before": validation_before,
                    "validation_after": validation_after,
                }
            )
    finally:
        runtime.stop()

    artifact = Path(args.artifact_dir)
    artifact.mkdir(parents=True, exist_ok=True)
    records_path = artifact / "records.jsonl"
    records_manifest = write_babysit_dataset(
        records_path,
        babysit_records,
        metadata={
            "curriculum": str(curriculum_path),
            "curriculum_sha256": sha256(curriculum_path),
            "cycles": len(curriculum["cycles"]),
            "on_policy": True,
            "teacher": "Arena.ai assistant under user direction",
            "hidden_reasoning_stored": False,
            "protected_evaluation_used": False,
        },
    )
    skills = {
        "schema_version": 1,
        "name": "aira-one-broad-skills-v1",
        "skills": [skill_payload(skill) for skill in installed],
        "source_records": str(records_path),
        "source_records_sha256": sha256(records_path),
    }
    skills_path = artifact / "skills.json"
    skills_path.write_text(
        json.dumps(skills, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    verdicts = Counter(record.verdict for record in babysit_records)
    summary = {
        "schema_version": 1,
        "model": "aira-one-v0.1",
        "experiment": "three-broad-on-policy-babysit-cycles-v1",
        "curriculum": str(curriculum_path),
        "curriculum_sha256": sha256(curriculum_path),
        "cycles": len(cycle_reports),
        "tasks": len(tasks),
        "train_attempts": len(babysit_records),
        "verdicts": dict(sorted(verdicts.items())),
        "skills_installed": len(installed),
        "validation_passes_before": sum(
            cycle["validation_passes_before"] for cycle in cycle_reports
        ),
        "validation_passes_after": sum(
            cycle["validation_passes_after"] for cycle in cycle_reports
        ),
        "validation_neural_calls_before": sum(
            sum(item["neural_calls"] for item in cycle["validation_before"])
            for cycle in cycle_reports
        ),
        "validation_neural_calls_after": sum(
            cycle["validation_neural_calls_after"] for cycle in cycle_reports
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "records_manifest": str(records_manifest),
        "skills": str(skills_path),
        "cycle_reports": cycle_reports,
        "limitations": (
            "Keyword concept rubrics and paired paraphrases measure installed skill coverage, "
            "not unrestricted general intelligence."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (artifact / "report.json").write_text(
        json.dumps(
            {key: value for key, value in summary.items() if key != "cycle_reports"},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (artifact / "README.md").write_text(
        "# AIra One broad Babysit v1\n\n"
        "Three broad cycles covering science, practical safety, and computing/AI. "
        "Only failed on-policy training attempts become high-confidence skill routes; "
        "paired paraphrases remain the validation side.\n\n"
        f"- Curriculum SHA-256: `{summary['curriculum_sha256']}`\n"
        f"- Student GGUF SHA-256: `{model_hash}`\n"
        f"- On-policy attempts: {summary['train_attempts']}\n"
        f"- Reviewed corrections installed: {summary['skills_installed']}\n"
        f"- Paired validation passes: {summary['validation_passes_before']}/24 -> "
        f"{summary['validation_passes_after']}/24\n"
        f"- Paired validation neural calls: {summary['validation_neural_calls_before']} "
        f"-> {summary['validation_neural_calls_after']}\n\n"
        "`records.jsonl` preserves each student attempt, deterministic verifier "
        "observations, reviewed teacher correction, model hash, and source prompt. "
        "`skills.json` is the auditable deployed shelf; its manifest binds every "
        "record hash in order. The teacher corrections were authored in the reviewed "
        "curriculum by the current competent assistant, not generated by the student.\n\n"
        "Concept-keyword rubrics and one paired paraphrase per topic provide regression "
        "evidence for these routes. They do not establish unrestricted semantic "
        "understanding or general intelligence.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "cycle_reports"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
