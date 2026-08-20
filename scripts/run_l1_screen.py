#!/usr/bin/env python3
"""Matched 5M-parameter real-data screen capped at 300 CPU training steps."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from pathlib import Path

import torch

from minillm.config import MiniLLMConfig
from minillm.training import TrainConfig, train_proxy


def clean_commit_hash() -> str:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"], text=True
    )
    if status:
        raise RuntimeError(
            "L1 screen requires a clean git worktree for exact provenance"
        )
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def first_validation(metrics_path: Path) -> float:
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if "validation_main_loss" in record:
            return float(record["validation_main_loss"])
    raise ValueError(f"no validation record in {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="data/tokens-github-pilot")
    parser.add_argument("--contract", default="configs/experiments/l1_real_screen.json")
    parser.add_argument(
        "--steps", type=int, default=300, choices=range(1, 301), metavar="[1-300]"
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--runs", default="runs/l1-real-screen")
    parser.add_argument("--output", default="results/l1_real_screen.json")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    seeds = args.seeds or contract["training"]["seeds"]
    if (
        args.steps != contract["training"]["steps"]
        or seeds != contract["training"]["seeds"]
    ):
        raise ValueError("screen command differs from preregistered steps or seeds")
    output = Path(args.output)
    runs = Path(args.runs)
    if output.exists() or runs.exists():
        raise FileExistsError("refusing to overwrite L1 screen outputs")
    torch.set_num_threads(args.threads)
    git_commit = clean_commit_hash()
    rows = []
    for variant, config_path in contract["variants"].items():
        model = MiniLLMConfig.load(config_path)
        for seed in seeds:
            run_directory = runs / f"{variant}-seed{seed}"
            training = TrainConfig(
                steps=args.steps,
                batch_size=contract["training"]["batch_size"],
                sequence_length=contract["training"]["sequence_length"],
                gradient_accumulation=contract["training"]["gradient_accumulation"],
                learning_rate=5e-4,
                warmup_steps=30,
                eval_interval=50,
                eval_batches=10,
                checkpoint_interval=args.steps,
                seed=seed,
                precision="fp32",
                schedule_tokens=contract["training"]["tokens_per_run"],
            )
            metadata = {
                "experiment": contract["id"],
                "variant": variant,
                "git_commit": git_commit,
                **contract["data"],
            }
            summary = train_proxy(
                model,
                training,
                train_tokens=Path(args.tokens) / "train.bin",
                validation_tokens=Path(args.tokens) / "validation.bin",
                output_directory=run_directory,
                run_metadata=metadata,
            )
            initial = first_validation(run_directory / "metrics.jsonl")
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "config": config_path,
                    "initial_validation_main_loss": initial,
                    "validation_improvement": initial
                    - summary["best_validation_main_loss"],
                    **summary,
                }
            )

    aggregates = []
    for variant in contract["variants"]:
        selected = [row for row in rows if row["variant"] == variant]
        losses = [row["best_validation_main_loss"] for row in selected]
        improvements = [row["validation_improvement"] for row in selected]
        aggregates.append(
            {
                "variant": variant,
                "runs": len(selected),
                "parameters": selected[0]["parameters"],
                "mean_best_validation_main_loss": statistics.mean(losses),
                "std_best_validation_main_loss": statistics.stdev(losses),
                "mean_validation_improvement": statistics.mean(improvements),
                "mean_tokens_per_second": statistics.mean(
                    row["tokens_per_second"] for row in selected
                ),
                "mean_wall_seconds": statistics.mean(
                    row["wall_seconds"] for row in selected
                ),
            }
        )
    by_variant = {row["variant"]: row for row in aggregates}
    gates = contract["primary_gates"]
    finite = all(
        math.isfinite(row["best_validation_main_loss"])
        and math.isfinite(row["last_validation_main_loss"])
        for row in rows
    )
    improvement_pass = all(
        row["validation_improvement"] >= gates["minimum_main_loss_improvement"]
        for row in rows
    )
    edge_ratio = (
        by_variant["edge"]["mean_best_validation_main_loss"]
        / by_variant["attention"]["mean_best_validation_main_loss"]
    )
    verdict = {
        "passed": finite
        and improvement_pass
        and edge_ratio <= gates["edge_mean_loss_ratio_max"],
        "all_runs_finite": finite,
        "all_runs_improve": improvement_pass,
        "edge_mean_loss_ratio": edge_ratio,
        "edge_ratio_limit": gates["edge_mean_loss_ratio_max"],
    }
    payload = {
        "contract": args.contract,
        "experiment": contract["id"],
        "git_commit": git_commit,
        "steps": args.steps,
        "seeds": seeds,
        "aggregates": aggregates,
        "runs": rows,
        "verdict": verdict,
        "warning": contract["non_claims"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
