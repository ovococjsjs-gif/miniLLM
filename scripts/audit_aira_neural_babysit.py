#!/usr/bin/env python3
"""Bind the explicit manual review to the neural Babysit experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--review", default="configs/aira-one/neural_babysit_manual_review_v1.json"
    )
    parser.add_argument("--raw", default="results/aira_neural_babysit_v1.json")
    parser.add_argument(
        "--output", default="results/aira_neural_babysit_v1_audited.json"
    )
    parser.add_argument(
        "--artifact-report",
        default="artifacts/aira-neural-babysit-v1/audited_report.json",
    )
    args = parser.parse_args()

    review_path = Path(args.review)
    raw_path = Path(args.raw)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    records = raw["free_generation"]["records"]
    record_by_id = {record["id"]: record for record in records}
    if len(record_by_id) != review["manual_totals"]["tasks"]:
        raise ValueError("manual review task count differs from raw evidence")

    manual: dict[str, dict[str, bool]] = {
        record["id"]: {
            side: bool(record[side]["verification"]["passed"])
            for side in ("base", "adapted")
        }
        for record in records
    }
    applied_overrides: list[dict[str, Any]] = []
    for override in review["overrides"]:
        record = record_by_id.get(override["id"])
        if record is None:
            raise ValueError(f"reviewed record is missing: {override['id']}")
        side = override["side"]
        automated = bool(record[side]["verification"]["passed"])
        if automated is not override["automated_passed"]:
            raise ValueError(f"automated verdict drifted for {override['id']}")
        manual[override["id"]][side] = bool(override["manual_passed"])
        applied_overrides.append(override)

    totals = {
        f"{side}_passes": sum(values[side] for values in manual.values())
        for side in ("base", "adapted")
    }
    expected = review["manual_totals"]
    if (
        totals["base_passes"] != expected["base_passes"]
        or totals["adapted_passes"] != expected["adapted_passes"]
    ):
        raise ValueError("manual total does not match reviewed decisions")
    controls = raw["out_of_scope_controls"]
    if (
        controls["byte_exact_preserved"]
        != expected["out_of_scope_byte_exact_preserved"]
        or controls["tasks"] != expected["out_of_scope_tasks"]
    ):
        raise ValueError("out-of-scope control total drifted")

    accepted = {
        side: sorted(identity for identity, values in manual.items() if values[side])
        for side in ("base", "adapted")
    }
    audited = {
        "schema_version": 1,
        "experiment": "aira-neural-babysit-output-adapter-v1-manually-audited",
        "raw_result": str(raw_path),
        "raw_result_sha256": sha256(raw_path),
        "manual_review": str(review_path),
        "manual_review_sha256": sha256(review_path),
        "architecture": {
            "base": "frozen Qwen3.5-0.8B Q4_K_M",
            "learned_component": "prompt-gated hidden-to-logit residual MLP",
            "parameters_changed": raw["training"]["parameters_changed"],
            "training_steps": raw["training"]["steps"],
            "stored_answer_routes_used": False,
            "keyword_routes_used": False,
            "adapter_checkpoint": raw["adapter"],
            "adapter_sha256": raw["adapter_sha256"],
        },
        "manual_quality": {
            "tasks": expected["tasks"],
            **totals,
            "absolute_improvement": totals["adapted_passes"] - totals["base_passes"],
            "accepted": accepted,
            "overrides": applied_overrides,
        },
        "teacher_forced_validation": raw["training"]["validation_teacher_forced"],
        "out_of_scope_controls": {
            "tasks": controls["tasks"],
            "byte_exact_preserved": controls["byte_exact_preserved"],
            "adapter_activations": controls["adapter_activations"],
            "failed_ids": [
                item["id"]
                for item in controls["records"]
                if not item["byte_exact_preserved"]
            ],
        },
        "decision": review["decision"],
        "rationale": review["rationale"],
        "limitations": review["limitations"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(audited, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    artifact_report = Path(args.artifact_report)
    artifact_report.parent.mkdir(parents=True, exist_ok=True)
    artifact_report.write_text(
        json.dumps(audited, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    artifact_root = artifact_report.parent
    manifest_path = artifact_root / "manifest.json"
    previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evidence_files = sorted(
        path
        for path in artifact_root.iterdir()
        if path.is_file() and path.name != manifest_path.name
    )
    previous_manifest["files"] = [
        {
            "path": str(path.relative_to(artifact_root)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in evidence_files
    ]
    previous_manifest["manual_review_sha256"] = sha256(review_path)
    manifest_path.write_text(
        json.dumps(previous_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audited, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
