#!/usr/bin/env python3
"""Compare matched Qwen donor evaluations without weakening strict acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

_COMPONENTS = ("strict", "content", "protocol", "source")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def compare(baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if baseline["dataset_sha256"] != candidate["dataset_sha256"]:
        raise ValueError("evaluations use different task datasets")
    baseline_samples = {sample["id"]: sample for sample in baseline["samples"]}
    candidate_samples = {sample["id"]: sample for sample in candidate["samples"]}
    if len(baseline_samples) != len(baseline["samples"]) or len(candidate_samples) != len(
        candidate["samples"]
    ):
        raise ValueError("evaluation sample IDs must be unique")
    if set(baseline_samples) != set(candidate_samples):
        raise ValueError("evaluations contain different task IDs")
    for identifier, baseline_sample in baseline_samples.items():
        candidate_sample = candidate_samples[identifier]
        if baseline_sample["category"] != candidate_sample["category"]:
            raise ValueError(f"category mismatch for task {identifier}")
        if baseline_sample["components"].get("source_required") != candidate_sample[
            "components"
        ].get("source_required"):
            raise ValueError(f"source applicability mismatch for task {identifier}")

    components: dict[str, Any] = {}
    changes: list[dict[str, Any]] = []
    for component in _COMPONENTS:
        applicable_ids = [
            identifier
            for identifier, sample in baseline_samples.items()
            if component != "source"
            or bool(sample["components"].get("source_required"))
        ]
        baseline_passed = sum(
            bool(baseline_samples[item]["components"][component])
            for item in applicable_ids
        )
        candidate_passed = sum(
            bool(candidate_samples[item]["components"][component])
            for item in applicable_ids
        )
        improved = [
            item
            for item in applicable_ids
            if not baseline_samples[item]["components"][component]
            and candidate_samples[item]["components"][component]
        ]
        regressed = [
            item
            for item in applicable_ids
            if baseline_samples[item]["components"][component]
            and not candidate_samples[item]["components"][component]
        ]
        components[component] = {
            "applicable": len(applicable_ids),
            "baseline_passed": baseline_passed,
            "candidate_passed": candidate_passed,
            "delta": candidate_passed - baseline_passed,
            "improved_ids": improved,
            "regressed_ids": regressed,
        }

    for identifier, baseline_sample in baseline_samples.items():
        candidate_sample = candidate_samples[identifier]
        changed = {
            component: {
                "baseline": bool(baseline_sample["components"][component]),
                "candidate": bool(candidate_sample["components"][component]),
            }
            for component in _COMPONENTS
            if baseline_sample["components"][component]
            != candidate_sample["components"][component]
        }
        if changed:
            changes.append(
                {
                    "id": identifier,
                    "category": baseline_sample["category"],
                    "components": changed,
                }
            )

    return {
        "schema_version": 1,
        "baseline": {
            "path": str(baseline_path),
            "sha256": sha256(baseline_path),
            "control_profile": baseline.get("control_profile", "baseline"),
        },
        "candidate": {
            "path": str(candidate_path),
            "sha256": sha256(candidate_path),
            "control_profile": candidate.get("control_profile", "baseline"),
        },
        "dataset_sha256": baseline["dataset_sha256"],
        "matched_tasks": len(baseline_samples),
        "components": components,
        "changed_tasks": changes,
        "conclusion": (
            "The protocol prompt is not accepted as a quality intervention unless strict "
            "passes improve without component regressions."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = compare(Path(args.baseline), Path(args.candidate))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
