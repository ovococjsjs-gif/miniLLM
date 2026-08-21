#!/usr/bin/env python3
"""Apply the explicit human audit and one independent remediation cycle."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minillm.aira.babysit import (
    BabysitRecord,
    VerifierObservation,
    read_babysit_dataset,
    write_babysit_dataset,
)
from minillm.aira.one import AIraOne, SkillShelf


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_answer(answer: str, task: dict[str, Any]) -> dict[str, Any]:
    lowered = answer.casefold()
    missing = [
        group
        for group in task["answer_requirements"]
        if not any(term.casefold() in lowered for term in group)
    ]
    forbidden = [
        term for term in task.get("forbidden", []) if term.casefold() in lowered
    ]
    return {
        "passed": bool(answer.strip()) and not missing and not forbidden,
        "nonempty": bool(answer.strip()),
        "missing_groups": missing,
        "forbidden_matches": forbidden,
    }


def phase_entries(raw: dict[str, Any], phase: str) -> list[dict[str, Any]]:
    key = {
        "train": "train_attempts",
        "validation_before": "validation_before",
        "validation_after": "validation_after",
    }[phase]
    return [entry for cycle in raw["cycle_reports"] for entry in cycle[key]]


def main() -> None:
    review_path = ROOT / "configs/aira-one/broad_manual_review_v1.json"
    curriculum_path = ROOT / "configs/aira-one/broad_curriculum_v1.json"
    raw_path = ROOT / "results/aira_one_broad_babysit_v1.json"
    artifact = ROOT / "artifacts/aira-one-broad-babysit-v1"
    original_records_path = artifact / "records.jsonl"
    audited_records_path = artifact / "records_audited.jsonl"
    skills_path = artifact / "skills.json"
    pre_review_skills_path = artifact / "skills_pre_review.json"
    output_path = ROOT / "results/aira_one_broad_babysit_v1_audited.json"

    review = json.loads(review_path.read_text(encoding="utf-8"))
    curriculum = json.loads(curriculum_path.read_text(encoding="utf-8"))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    tasks = {
        task["id"]: task for cycle in curriculum["cycles"] for task in cycle["tasks"]
    }
    if len(tasks) != 24 or raw["tasks"] != 24:
        raise ValueError("manual review is pinned to exactly 24 curriculum tasks")

    for override in review["overrides"]:
        entries = phase_entries(raw, override["phase"])
        entry = next(
            (item for item in entries if item["task_id"] == override["task_id"]),
            None,
        )
        if entry is None:
            raise ValueError(f"missing reviewed entry: {override}")
        if entry["verification"]["passed"] is not override["automated_passed"]:
            raise ValueError(f"automated verdict drifted for {override['task_id']}")

    if not pre_review_skills_path.exists():
        pre_review_skills_path.write_bytes(skills_path.read_bytes())
    pre_review_skills = json.loads(pre_review_skills_path.read_text(encoding="utf-8"))
    if len(pre_review_skills["skills"]) != 23:
        raise ValueError("pre-review shelf must contain the 23 generated corrections")

    remediation = review["remediation"]
    task = tasks[remediation["task_id"]]
    source = next(
        entry
        for entry in phase_entries(raw, remediation["source_phase"])
        if entry["task_id"] == remediation["task_id"]
    )
    if source["prompt"] != remediation["source_prompt"]:
        raise ValueError("remediation source prompt drifted")
    wrong_example = source["answer"].find("100°C")
    correction = task["answers"]["ru"]
    records = read_babysit_dataset(original_records_path)
    if len(records) != 24:
        raise ValueError("raw broad dataset must contain 24 on-policy attempts")
    records.append(
        BabysitRecord(
            task_id="aira-one-broad:manual-remediation:boiling-pressure",
            prompt=source["prompt"],
            student_checkpoint=records[0].student_checkpoint,
            teacher_id="arena-agent-teacher-v1",
            student_answer=source["answer"],
            verifier_observations=(
                VerifierObservation(
                    tool="aira-broad-manual-review-v1",
                    passed=False,
                    detail=(
                        "Material factual contradiction: the answer says water at "
                        "3000 m boils near 100 C and misstates the altitude/pressure "
                        "example, despite containing the rubric keywords."
                    ),
                ),
            ),
            verdict="incorrect",
            critique=(
                "The qualitative opening is correct, but the numerical examples reverse "
                "or erase the altitude effect. Lower external pressure lowers the boiling "
                "temperature."
            ),
            corrected_answer=correction,
            rubric_scores={"accuracy": 0.0, "relevance": 0.7, "safety": 1.0},
            teacher_confidence=0.99,
            first_error_offset=max(wrong_example, 0),
            error_type="manual-review-factual-contradiction",
        )
    )
    audited_manifest_path = write_babysit_dataset(
        audited_records_path,
        records,
        metadata={
            "curriculum": str(curriculum_path.relative_to(ROOT)),
            "curriculum_sha256": sha256(curriculum_path),
            "manual_review": str(review_path.relative_to(ROOT)),
            "manual_review_sha256": sha256(review_path),
            "cycles": 4,
            "on_policy": True,
            "teacher": "Arena.ai assistant under user direction",
            "hidden_reasoning_stored": False,
            "protected_evaluation_used": False,
        },
    )
    audited_manifest = json.loads(audited_manifest_path.read_text(encoding="utf-8"))

    remediation_skill = {
        "skill_id": f"science.{task['id']}",
        "required_groups": task["required_groups"],
        "answers": task["answers"],
        "provenance": "aira-one-broad-manual-review-v1",
    }
    skills = list(pre_review_skills["skills"])
    if any(item["skill_id"] == remediation_skill["skill_id"] for item in skills):
        raise ValueError("remediation skill unexpectedly exists in pre-review shelf")
    skills.append(remediation_skill)
    shelf_payload = {
        "schema_version": 1,
        "name": "AIra One broad Babysit shelf v1, manually audited",
        "skills": skills,
        "source_records": len(records),
        "source_records_sha256": audited_manifest["sha256"],
        "manual_review": str(review_path.relative_to(ROOT)),
        "manual_review_sha256": sha256(review_path),
    }
    skills_path.write_text(
        json.dumps(shelf_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    assistant = AIraOne(None, skill_shelf=SkillShelf.load(skills_path))
    independent_results = []
    for task_id, item in tasks.items():
        prompt = (
            remediation["independent_validation_prompt"]
            if task_id == remediation["task_id"]
            else item["validation_prompt"]
        )
        response = assistant.answer(prompt)
        check = verify_answer(response.answer, item)
        independent_results.append(
            {
                "task_id": task_id,
                "prompt": prompt,
                "answer": response.answer,
                "route": response.route,
                "model_bypassed": response.model_bypassed,
                "neural_calls": response.neural_calls,
                "latency_seconds": response.latency_seconds,
                "verification": check,
            }
        )
    if not all(
        item["verification"]["passed"]
        and item["model_bypassed"]
        and item["neural_calls"] == 0
        for item in independent_results
    ):
        raise RuntimeError(
            "audited independent regression set did not fully bypass/pass"
        )

    raw_cycles = {cycle["cycle"]: cycle for cycle in raw["cycle_reports"]}
    manual_cycles = []
    for manual in review["cycle_totals"]:
        raw_cycle = raw_cycles[manual["cycle"]]
        manual_cycles.append(
            {
                **manual,
                "tasks": raw_cycle["tasks"],
                "validation_neural_calls_before": sum(
                    item["neural_calls"] for item in raw_cycle["validation_before"]
                ),
                "validation_neural_calls_after": sum(
                    item["neural_calls"] for item in raw_cycle["validation_after"]
                ),
                "validation_request_latency_seconds_before": sum(
                    item["latency_seconds"] for item in raw_cycle["validation_before"]
                ),
                "validation_request_latency_seconds_after": sum(
                    item["latency_seconds"] for item in raw_cycle["validation_after"]
                ),
                "skills_installed_by_automated_gate": raw_cycle["skills_installed"],
            }
        )
    manual_cycles.append(
        {
            "cycle": remediation["cycle"],
            "tasks": 1,
            "train_passes": 0,
            "validation_passes_before": 0,
            "validation_passes_after": 1,
            "validation_neural_calls_before": source["neural_calls"],
            "validation_neural_calls_after": 0,
            "validation_request_latency_seconds_before": source["latency_seconds"],
            "validation_request_latency_seconds_after": next(
                item["latency_seconds"]
                for item in independent_results
                if item["task_id"] == remediation["task_id"]
            ),
            "skills_installed_by_manual_gate": 1,
        }
    )

    audited = {
        "schema_version": 1,
        "model": "aira-one-v0.1",
        "experiment": "broad-on-policy-babysit-v1-manually-audited",
        "raw_result": str(raw_path.relative_to(ROOT)),
        "raw_result_sha256": sha256(raw_path),
        "manual_review": str(review_path.relative_to(ROOT)),
        "manual_review_sha256": sha256(review_path),
        "curriculum_sha256": sha256(curriculum_path),
        "student_checkpoint": records[0].student_checkpoint,
        "automated_three_cycle": {
            "train_passes": raw["verdicts"]["correct"],
            "validation_passes_before": raw["validation_passes_before"],
            "validation_passes_after": raw["validation_passes_after"],
            "validation_neural_calls_before": raw["validation_neural_calls_before"],
            "validation_neural_calls_after": raw["validation_neural_calls_after"],
            "skills_installed": raw["skills_installed"],
        },
        "manual_three_cycle": {
            **review["manual_totals"],
            "validation_neural_calls_before": raw["validation_neural_calls_before"],
            "validation_neural_calls_after": raw["validation_neural_calls_after"],
            "reviewed_answers": sum(review["coverage"].values())
            - int(review["coverage"]["hidden_reasoning_requested_or_stored"]),
            "false_negatives_corrected": 2,
            "false_positives_corrected": 3,
        },
        "cycle_reports": manual_cycles,
        "remediation": {
            "source_task_id": remediation["task_id"],
            "source_prompt": source["prompt"],
            "source_answer": source["answer"],
            "source_answer_sha256": hashlib.sha256(
                source["answer"].encode("utf-8")
            ).hexdigest(),
            "manual_verdict": "incorrect",
            "teacher_correction": correction,
            "independent_validation": next(
                item
                for item in independent_results
                if item["task_id"] == remediation["task_id"]
            ),
        },
        "final_independent_regression": {
            "tasks": 24,
            "passes": 24,
            "neural_calls": 0,
            "bypassed_requests": 24,
            "skills_installed": len(skills),
            "total_request_latency_seconds": sum(
                item["latency_seconds"] for item in independent_results
            ),
            "mean_request_latency_seconds": sum(
                item["latency_seconds"] for item in independent_results
            )
            / len(independent_results),
            "baseline_total_request_latency_seconds": sum(
                item["latency_seconds"]
                for cycle in raw["cycle_reports"]
                for item in cycle["validation_before"]
            ),
            "after_three_cycles_total_request_latency_seconds": sum(
                item["latency_seconds"]
                for cycle in raw["cycle_reports"]
                for item in cycle["validation_after"]
            ),
            "latency_scope": (
                "Summed AIraOne.answer request latency on this CPU run; final set uses "
                "the independent remediation paraphrase for boiling-pressure."
            ),
            "results": independent_results,
        },
        "evidence": {
            "raw_records": str(original_records_path.relative_to(ROOT)),
            "audited_records": str(audited_records_path.relative_to(ROOT)),
            "audited_records_sha256": audited_manifest["sha256"],
            "skills": str(skills_path.relative_to(ROOT)),
            "skills_sha256": sha256(skills_path),
        },
        "limitations": review["limitations"],
    }
    output_path.write_text(
        json.dumps(audited, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (artifact / "audited_report.json").write_text(
        json.dumps(
            {
                key: value
                for key, value in audited.items()
                if key != "final_independent_regression"
            }
            | {
                "final_independent_regression": {
                    key: value
                    for key, value in audited["final_independent_regression"].items()
                    if key != "results"
                }
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (artifact / "README.md").write_text(
        "# AIra One broad Babysit v1\n\n"
        "Three broad cycles cover science, practical safety, and computing/AI; one "
        "manual-audit remediation cycle corrects a keyword-verifier false positive. "
        "Only reviewed on-policy failures become high-confidence skill routes, and "
        "every correction is checked on a separate paraphrase.\n\n"
        f"- Curriculum SHA-256: `{sha256(curriculum_path)}`\n"
        f"- Student GGUF SHA-256: `{records[0].student_checkpoint}`\n"
        "- Initial on-policy training attempts: 24\n"
        "- Initial validation Qwen calls: 24\n"
        "- Automated concept result: 4/24 -> 24/24, 23 installed routes\n"
        "- Manually reviewed result after three cycles: 2/24 -> 23/24\n"
        "- Final result after one remediation cycle: 24/24 independent routes, "
        "0 Qwen calls, 24 installed routes\n"
        f"- Summed validation request latency on this CPU run: "
        f"{audited['final_independent_regression']['baseline_total_request_latency_seconds']:.2f} s "
        f"-> {audited['final_independent_regression']['after_three_cycles_total_request_latency_seconds']:.2f} s "
        f"-> {audited['final_independent_regression']['total_request_latency_seconds']:.4f} s\n\n"
        "`records.jsonl` and its manifest are the immutable raw three-cycle evidence. "
        "`records_audited.jsonl` adds the manually detected boiling-pressure failure "
        "and its teacher correction. `skills_pre_review.json` preserves the generated "
        "23-route shelf; `skills.json` is the final audited 24-route shelf. "
        "`audited_report.json` summarizes the manual decisions and final regression.\n\n"
        "Each record preserves the student prompt and answer, verifier observations, "
        "reviewed teacher correction, model hash, text hashes, and provenance. Teacher "
        "corrections were authored in the reviewed curriculum by the current competent "
        "assistant, not generated by the student. No hidden reasoning is stored.\n\n"
        "Concept-keyword rubrics, explicit manual review, and one held-out paraphrase "
        "per installed correction provide bounded regression evidence. They do not "
        "establish unrestricted semantic understanding or general intelligence.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manual_three_cycle": audited["manual_three_cycle"],
                "cycle_reports": manual_cycles,
                "final_independent_regression": {
                    key: value
                    for key, value in audited["final_independent_regression"].items()
                    if key != "results"
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
