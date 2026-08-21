#!/usr/bin/env python3
"""Measure a bounded CPU smoke for the pinned Qwen3.5 donor."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import resource
import subprocess
import time
from pathlib import Path

_SPEED = re.compile(r"Prompt: ([0-9.]+) t/s \| Generation: ([0-9.]+) t/s")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/donors/qwen35_08b.json")
    parser.add_argument(
        "--model",
        default=(
            "data/external/qwen3.5-0.8b/"
            "Qwen3.5-0.8B-Q4_K_M-github.gguf"
        ),
    )
    parser.add_argument(
        "--llama-cli", default=".cache/llama.cpp/build/bin/llama-cli"
    )
    parser.add_argument("--output", default="results/qwen35_08b_runtime_smoke.json")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = Path(args.model)
    mirror = config["github_mirror"]
    if model.stat().st_size != mirror["size_bytes"] or sha256(model) != mirror["sha256"]:
        raise ValueError("Qwen donor differs from pinned GitHub mirror")

    command = [
        args.llama_cli,
        "--model",
        str(model),
        "--ctx-size",
        "2048",
        "--threads",
        "2",
        "--threads-batch",
        "2",
        "--batch-size",
        "128",
        "--ubatch-size",
        "128",
        "--predict",
        "96",
        "--system-prompt",
        "Ты точный русско-английский помощник. Проверяй числа и отвечай кратко.",
        "--prompt",
        "У Маши было 17 яблок, она купила ещё 8 и отдала 6. Сколько яблок осталось?",
        "--conversation",
        "--single-turn",
        "--reasoning",
        "off",
        "--temp",
        "0",
        "--seed",
        "42",
        "--no-display-prompt",
        "--simple-io",
        "--show-timings",
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command, check=False, text=True, capture_output=True, timeout=300
    )
    elapsed = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(completed.stderr[-2000:])
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    speed = _SPEED.search(completed.stdout)
    correct_answer_present = "= 19" in completed.stdout
    if not correct_answer_present:
        raise RuntimeError("bounded arithmetic smoke did not contain the expected answer")
    report = {
        "schema_version": 1,
        "role": "language-donor-and-control-not-teacher",
        "model": str(model),
        "model_size_bytes": model.stat().st_size,
        "model_sha256": sha256(model),
        "llama_cpp_revision": config["runtime"]["revision"],
        "context_length": 2048,
        "threads": 2,
        "batch_size": 128,
        "elapsed_seconds": elapsed,
        "child_maxrss_kib": usage.ru_maxrss,
        "prompt_tokens_per_second": float(speed.group(1)) if speed else None,
        "generation_tokens_per_second": float(speed.group(2)) if speed else None,
        "correct_answer_present": correct_answer_present,
        "transcript": completed.stdout,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "transcript"}, indent=2))


if __name__ == "__main__":
    main()
