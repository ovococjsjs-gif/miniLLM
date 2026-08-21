#!/usr/bin/env python3
"""Evaluate a local Qwen donor server with the strict AIra Mentor verifiers."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from minillm.aira.provider import OpenAIChatProvider, ProviderError
from minillm.aira.verification import verify_synthetic_generation


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--provider-model", default="qwen3.5-0.8b-q4-k-m")
    parser.add_argument(
        "--dataset", default="artifacts/aira-mentor-v1/test.jsonl"
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output", default="results/qwen35_08b_donor_baseline.json")
    args = parser.parse_args()
    if args.limit < 0:
        raise ValueError("limit cannot be negative")
    dataset = Path(args.dataset)
    records = load_records(dataset)
    if args.limit:
        records = records[: args.limit]
    provider = OpenAIChatProvider(
        base_url=args.endpoint,
        model=args.provider_model,
        timeout_seconds=args.timeout,
    )
    category_totals: Counter[str] = Counter()
    category_passes: Counter[str] = Counter()
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
        passed = verify_synthetic_generation(record, response.content)
        category_totals[record["category"]] += 1
        category_passes[record["category"]] += passed
        samples.append(
            {
                "id": record["id"],
                "category": record["category"],
                "language": record["language"],
                "passed": passed,
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
        "requested_records": len(records),
        "completed_records": len(samples),
        "errors": errors,
        "passed": passed_total,
        "pass_rate": passed_total / len(samples) if samples else 0.0,
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
