#!/usr/bin/env python3
"""Capacity/noise/runtime probe for the bounded AIra-v2 associative memory."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from minillm.aira import BoundedAssociativeMemory


def noisy(code: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    result = code.copy()
    count = round(len(code) * fraction)
    if count:
        positions = rng.choice(len(code), count, replace=False)
        result[positions] *= -1
    return result


def run_scale(
    n: int,
    *,
    dimension: int,
    queries: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    codes = rng.choice(np.int8([-1, 1]), size=(n, dimension))
    memory = BoundedAssociativeMemory(
        capacity=n,
        dimension=dimension,
        similarity_threshold=0.3,
        margin_threshold=0.02,
    )
    for index, code in enumerate(codes):
        memory.write(code, index, provenance={"source": "generated", "index": index})

    selected = rng.choice(n, size=min(queries, n), replace=False)
    noise_results = {}
    for fraction in (0.0, 0.1, 0.2, 0.3, 0.4):
        accepted = correct = 0
        started = time.perf_counter()
        for index in selected:
            hit = memory.query(noisy(codes[index], fraction, rng))
            accepted += int(hit.accepted)
            correct += int(hit.accepted and hit.payload == int(index))
        elapsed = time.perf_counter() - started
        noise_results[str(fraction)] = {
            "accepted": accepted,
            "correct": correct,
            "recall": correct / len(selected),
            "precision": correct / accepted if accepted else None,
            "mean_query_milliseconds": 1000 * elapsed / len(selected),
        }

    unknown_accepted = 0
    started = time.perf_counter()
    for _ in range(len(selected)):
        query = rng.choice(np.int8([-1, 1]), size=dimension)
        unknown_accepted += int(memory.query(query).accepted)
    unknown_seconds = time.perf_counter() - started
    return {
        "facts": n,
        "dimension": dimension,
        "code_storage_bytes": memory.code_storage_bytes,
        "scan_operations_per_query": memory.scan_operations(),
        "noise": noise_results,
        "unknown_queries": len(selected),
        "unknown_accept_rate": unknown_accepted / len(selected),
        "unknown_mean_query_milliseconds": 1000 * unknown_seconds / len(selected),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts", nargs="+", type=int, default=[100, 1000, 5000])
    parser.add_argument("--dimension", type=int, default=512)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/aira_memory_proxy.json")
    args = parser.parse_args()
    rows = [
        run_scale(
            facts,
            dimension=args.dimension,
            queries=args.queries,
            seed=args.seed + facts,
        )
        for facts in args.facts
    ]
    payload = {
        "schema_version": 1,
        "experiment": "aira-v2-bounded-associative-memory-v1",
        "warning": "Random bipolar-code reference; semantic encoding and ANN indexing are not established.",
        "implementation": "one-shot ring-bounded storage; exact O(ND) scan; familiarity+margin gate",
        "platform": platform.platform(),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
