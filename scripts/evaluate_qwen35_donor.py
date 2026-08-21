#!/usr/bin/env python3
"""Evaluate a local Qwen donor server with the strict AIra Mentor verifiers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any


def load_lightweight_module(name: str, relative_path: str) -> ModuleType:
    """Load stdlib-only donor helpers without importing the Torch AIra package."""

    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_provider = load_lightweight_module(
    "minillm_lightweight_provider", "src/minillm/aira/provider.py"
)
_verification = load_lightweight_module(
    "minillm_lightweight_verification", "src/minillm/aira/verification.py"
)
OpenAIChatProvider = _provider.OpenAIChatProvider
ProviderError = _provider.ProviderError
synthetic_generation_components = _verification.synthetic_generation_components
verify_synthetic_generation = _verification.verify_synthetic_generation


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_records(
    records: list[dict[str, Any]],
    *,
    limit: int = 0,
    examples_per_category: int = 0,
) -> list[dict[str, Any]]:
    """Select a bounded prefix or a deterministic category-balanced subset."""

    if limit < 0 or examples_per_category < 0:
        raise ValueError("evaluation limits cannot be negative")
    if limit and examples_per_category:
        raise ValueError("choose either a global limit or a per-category limit")
    if examples_per_category:
        selected: list[dict[str, Any]] = []
        selected_counts: Counter[str] = Counter()
        for record in records:
            category = record["category"]
            if selected_counts[category] < examples_per_category:
                selected.append(record)
                selected_counts[category] += 1
        category_counts = Counter(record["category"] for record in records)
        missing = {
            category: examples_per_category - selected_counts[category]
            for category in category_counts
            if selected_counts[category] < examples_per_category
        }
        if missing:
            raise ValueError(f"dataset cannot satisfy per-category selection: {missing}")
        return selected
    if limit:
        return records[:limit]
    return list(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--provider-model", default="qwen3.5-0.8b-q4-k-m")
    parser.add_argument(
        "--dataset", default="artifacts/aira-mentor-v1/test.jsonl"
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--examples-per-category", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output", default="results/qwen35_08b_donor_baseline.json")
    args = parser.parse_args()
    dataset = Path(args.dataset)
    all_records = load_records(dataset)
    records = select_records(
        all_records,
        limit=args.limit,
        examples_per_category=args.examples_per_category,
    )
    provider = OpenAIChatProvider(
        base_url=args.endpoint,
        model=args.provider_model,
        timeout_seconds=args.timeout,
    )
    category_totals: Counter[str] = Counter()
    category_passes: Counter[str] = Counter()
    component_passes: Counter[str] = Counter()
    component_applicable: Counter[str] = Counter()
    samples = []
    errors = []
    started = time.perf_counter()
    for index, record in enumerate(records):
        messages = [
            dict(message)
            for message in record["messages"]
            if message["role"] != "assistant"
        ]
        request_started = time.perf_counter()
        try:
            response = provider.complete(
                messages, temperature=0, max_tokens=args.max_tokens
            )
        except ProviderError as error:
            errors.append(
                {
                    "id": record["id"],
                    "category": record["category"],
                    "error": str(error),
                }
            )
            continue
        latency = time.perf_counter() - request_started
        components = synthetic_generation_components(record, response.content)
        passed = components["strict"]
        category_totals[record["category"]] += 1
        category_passes[record["category"]] += passed
        for component in ("strict", "content", "protocol"):
            component_applicable[component] += 1
            component_passes[component] += components[component]
        if components["source_required"]:
            component_applicable["source"] += 1
            component_passes["source"] += components["source"]
        samples.append(
            {
                "id": record["id"],
                "category": record["category"],
                "language": record["language"],
                "passed": passed,
                "components": components,
                "answer": response.content,
                "answer_sha256": hashlib.sha256(response.content.encode()).hexdigest(),
                "reasoning_content_present": bool(response.reasoning_content),
                "finish_reason": response.finish_reason,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "latency_seconds": latency,
            }
        )
        print(
            f"[{index + 1}/{len(records)}] {record['category']} "
            f"{'PASS' if passed else 'FAIL'} {latency:.2f}s"
        )
    elapsed = time.perf_counter() - started
    passed_total = sum(category_passes.values())
    report = {
        "schema_version": 1,
        "role": "language-donor-and-control-not-teacher",
        "endpoint": args.endpoint,
        "provider_model": args.provider_model,
        "dataset": str(dataset),
        "dataset_sha256": sha256(dataset),
        "dataset_records": len(all_records),
        "selection": {
            "global_limit": args.limit,
            "examples_per_category": args.examples_per_category,
        },
        "requested_records": len(records),
        "completed_records": len(samples),
        "errors": errors,
        "passed": passed_total,
        "pass_rate": passed_total / len(samples) if samples else 0.0,
        "component_passes": dict(sorted(component_passes.items())),
        "component_applicable": dict(sorted(component_applicable.items())),
        "component_pass_rates": {
            key: component_passes[key] / applicable if applicable else 0.0
            for key, applicable in sorted(component_applicable.items())
        },
        "category_totals": dict(sorted(category_totals.items())),
        "category_passes": dict(sorted(category_passes.items())),
        "elapsed_seconds": elapsed,
        "samples": samples,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
