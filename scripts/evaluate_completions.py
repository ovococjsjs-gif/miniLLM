#!/usr/bin/env python3
"""Run a fixed bilingual completion suite against a trusted checkpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from minillm.evaluation import evaluate_completion_suite, load_completion_cases
from minillm.generation import SamplingConfig, load_model_checkpoint
from minillm.tokenization import load_tokenizer


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--config")
    parser.add_argument("--suite", default="eval/bilingual_smoke.json")
    parser.add_argument("--output", default="results/completion_smoke.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recurrences", type=int)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-bos", action="store_true")
    args = parser.parse_args()

    loaded = load_model_checkpoint(
        args.checkpoint, config_path=args.config, device=args.device
    )
    tokenizer = load_tokenizer(args.tokenizer)
    if tokenizer.get_vocab_size() != loaded.config.vocab_size:
        raise ValueError("tokenizer vocabulary does not match model configuration")
    cases = load_completion_cases(args.suite)
    report = evaluate_completion_suite(
        loaded.model,
        tokenizer,
        cases,
        SamplingConfig(
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
            use_cache=not args.no_cache,
        ),
        add_bos=not args.no_bos,
        core_repetitions=args.recurrences,
    )
    report.update(
        {
            "checkpoint": args.checkpoint,
            "checkpoint_step": loaded.step,
            "checkpoint_format": loaded.format_version,
            "tokenizer": args.tokenizer,
            "suite": args.suite,
            "git_commit": git_commit(),
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
