#!/usr/bin/env python3
"""Re-run current AIra component verifiers over stored donor generations."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any


def load_lightweight_module(name: str, relative_path: str) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_verification = load_lightweight_module(
    "minillm_rescore_verification", "src/minillm/aira/verification.py"
)
synthetic_generation_components = _verification.synthetic_generation_components


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {record["id"]: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("dataset task IDs must be unique")
    return by_id


def rescore(evaluation_path: Path, dataset_path: Path) -> dict[str, Any]:
    report = json.loads(evaluation_path.read_text(encoding="utf-8"))
    records = load_records(dataset_path)
    samples = report["samples"]
    if len({sample["id"] for sample in samples}) != len(samples):
        raise ValueError("evaluation sample IDs must be unique")
    missing = {sample["id"] for sample in samples} - set(records)
    if missing:
        raise ValueError(f"evaluation samples missing from dataset: {sorted(missing)}")

    category_totals: Counter[str] = Counter()
    category_passes: Counter[str] = Counter()
    component_passes: Counter[str] = Counter()
    component_applicable: Counter[str] = Counter()
    for sample in samples:
        record = records[sample["id"]]
        components = synthetic_generation_components(record, sample["answer"])
        sample["components"] = components
        sample["passed"] = components["strict"]
        category_totals[record["category"]] += 1
        category_passes[record["category"]] += components["strict"]
        for component in ("strict", "content", "protocol"):
            component_applicable[component] += 1
            component_passes[component] += components[component]
        if components["source_required"]:
            component_applicable["source"] += 1
            component_passes["source"] += components["source"]

    passed = component_passes["strict"]
    report["passed"] = passed
    report["pass_rate"] = passed / len(samples) if samples else 0.0
    report["component_passes"] = dict(sorted(component_passes.items()))
    report["component_applicable"] = dict(sorted(component_applicable.items()))
    report["component_pass_rates"] = {
        key: component_passes[key] / applicable if applicable else 0.0
        for key, applicable in sorted(component_applicable.items())
    }
    report["category_totals"] = dict(sorted(category_totals.items()))
    report["category_passes"] = dict(sorted(category_passes.items()))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    evaluation_path = Path(args.evaluation)
    output = Path(args.output) if args.output else evaluation_path
    report = rescore(evaluation_path, Path(args.dataset))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "evaluation": str(output),
                "passed": report["passed"],
                "component_passes": report["component_passes"],
                "component_applicable": report["component_applicable"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
