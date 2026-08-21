#!/usr/bin/env python3
"""Evaluate AIra One exact/residual routing with strict Mentor verifiers."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from minillm.aira import AIraMode, AIraOne, OpenAIChatProvider
from minillm.aira.synthetic import generate_aira_mentor_records
from minillm.aira.verification import synthetic_generation_components


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="artifacts/aira-mentor-v1/test.jsonl")
    parser.add_argument("--fresh-seed", type=int, default=0)
    parser.add_argument("--fresh-examples-per-category", type=int, default=10)
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--mode", choices=[item.value for item in AIraMode], default="balanced")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="results/aira_one_v01_evaluation.json")
    args = parser.parse_args()

    if args.fresh_seed:
        records = [
            record.to_dict()
            for record in generate_aira_mentor_records(
                examples_per_category=args.fresh_examples_per_category,
                seed=args.fresh_seed,
            )
        ]
        dataset_name = f"generated-seed-{args.fresh_seed}"
        dataset_hash = hashlib.sha256(
            "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True)
                for record in records
            ).encode()
        ).hexdigest()
        protected = False
    else:
        dataset_path = Path(args.dataset)
        records = load_jsonl(dataset_path)
        dataset_name = str(dataset_path)
        dataset_hash = sha256(dataset_path)
        protected = dataset_path.name in {"validation.jsonl", "test.jsonl"}
    if args.limit:
        records = records[: args.limit]

    provider = (
        OpenAIChatProvider(
            base_url=args.endpoint,
            model="aira-one-donor",
            timeout_seconds=240,
        )
        if args.endpoint
        else None
    )
    assistant = AIraOne(provider)
    component_passes: Counter[str] = Counter()
    component_applicable: Counter[str] = Counter()
    category_totals: Counter[str] = Counter()
    category_passes: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    samples = []
    started = time.perf_counter()
    for record in records:
        system = next(
            message["content"]
            for message in record["messages"]
            if message["role"] == "system"
        )
        user = next(
            message["content"]
            for message in record["messages"]
            if message["role"] == "user"
        )
        response = assistant.answer(
            user,
            system_text=system,
            mode=args.mode,
        )
        components = synthetic_generation_components(record, response.answer)
        category = record["category"]
        category_totals[category] += 1
        category_passes[category] += components["strict"]
        route_counts[response.route] += 1
        for component in ("strict", "content", "protocol"):
            component_applicable[component] += 1
            component_passes[component] += components[component]
        if components["source_required"]:
            component_applicable["source"] += 1
            component_passes["source"] += components["source"]
        samples.append(
            {
                "id": record["id"],
                "category": category,
                "language": record["language"],
                "answer": response.answer,
                "answer_sha256": hashlib.sha256(response.answer.encode()).hexdigest(),
                "components": components,
                "route": response.route,
                "model_bypassed": response.model_bypassed,
                "neural_calls": response.neural_calls,
                "latency_seconds": response.latency_seconds,
            }
        )

    completed = len(samples)
    report = {
        "schema_version": 1,
        "model": "aira-one-v0.1",
        "scope": (
            "deterministic Mentor-family contract and routing control; not a general "
            "assistant intelligence benchmark"
        ),
        "dataset": dataset_name,
        "dataset_sha256": dataset_hash,
        "protected_evaluation": protected,
        "records": completed,
        "mode": args.mode,
        "provider_enabled": provider is not None,
        "strict_passes": component_passes["strict"],
        "strict_rate": component_passes["strict"] / completed if completed else 0.0,
        "component_passes": dict(sorted(component_passes.items())),
        "component_applicable": dict(sorted(component_applicable.items())),
        "component_rates": {
            key: component_passes[key] / count if count else 0.0
            for key, count in sorted(component_applicable.items())
        },
        "category_totals": dict(sorted(category_totals.items())),
        "category_passes": dict(sorted(category_passes.items())),
        "route_counts": dict(sorted(route_counts.items())),
        "bypassed_records": sum(sample["model_bypassed"] for sample in samples),
        "neural_calls": sum(sample["neural_calls"] for sample in samples),
        "elapsed_seconds": time.perf_counter() - started,
        "runtime_stats": assistant.stats.to_dict(),
        "samples": samples,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "samples"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
