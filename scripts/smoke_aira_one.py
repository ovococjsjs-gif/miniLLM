#!/usr/bin/env python3
"""Run a mixed exact/neural AIra One v0.1 local smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

from minillm.aira import AIraMode, AIraOne, LocalDonorRuntime, OpenAIChatProvider


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=(
            "data/external/qwen3.5-0.8b/"
            "Qwen3.5-0.8B-Q4_K_M-github.gguf"
        ),
    )
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument(
        "--output", default="results/aira_one_v01_runtime_smoke.json"
    )
    args = parser.parse_args()

    runtime = LocalDonorRuntime(
        model=args.model,
        port=args.port,
        log_path=".aira-one/runtime-smoke-llama.log",
    )
    startup_started = time.perf_counter()
    runtime.start()
    startup_seconds = time.perf_counter() - startup_started
    provider = OpenAIChatProvider(
        base_url=runtime.endpoint,
        model="aira-one-donor",
        timeout_seconds=240,
    )
    assistant = AIraOne(provider)
    requests = [
        ("Вычисли: (37 + 18 - 9) * 3", AIraMode.FAST),
        (
            (
                "Документ doc-demo: проект Project-42 создан в 2020 году в городе Pskov. "
                "Цвет эмблемы — синий.\n\nВ каком городе создан проект Project-42?"
            ),
            AIraMode.FAST,
        ),
        (
            "Кратко объясни простыми словами, почему небо кажется голубым.",
            AIraMode.BALANCED,
        ),
        (
            "Give two practical tips for keeping a local computer secure.",
            AIraMode.FAST,
        ),
        (
            "Сравни локальную и облачную модель: назови по два плюса каждой.",
            AIraMode.DEEP,
        ),
    ]
    samples = []
    try:
        for prompt, mode in requests:
            response = assistant.answer(prompt, mode=mode)
            samples.append(
                {
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "mode": mode.value,
                    "answer": response.answer,
                    "answer_sha256": hashlib.sha256(response.answer.encode()).hexdigest(),
                    "route": response.route,
                    "model_bypassed": response.model_bypassed,
                    "neural_calls": response.neural_calls,
                    "latency_seconds": response.latency_seconds,
                    "citations": list(response.citations),
                    "verifier": dict(response.verifier),
                }
            )
    finally:
        runtime.stop()
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    if samples[0]["answer"] != "Ответ: 138.":
        raise RuntimeError("exact arithmetic route failed")
    if "[doc-demo]" not in samples[1]["answer"]:
        raise RuntimeError("inline document route lost its citation")
    if any(not sample["answer"].strip() for sample in samples):
        raise RuntimeError("AIra One produced an empty answer")

    report = {
        "schema_version": 1,
        "model": "aira-one-v0.1",
        "donor_model": str(args.model),
        "donor_sha256": sha256(Path(args.model)),
        "startup_seconds": startup_seconds,
        "child_maxrss_kib": usage.ru_maxrss,
        "requests": len(samples),
        "bypassed_requests": sum(sample["model_bypassed"] for sample in samples),
        "neural_calls": sum(sample["neural_calls"] for sample in samples),
        "runtime_stats": assistant.stats.to_dict(),
        "samples": samples,
        "interpretation": (
            "Mixed route/runtime smoke only. Exact-route success and non-empty neural "
            "answers do not establish broad assistant quality."
        ),
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
