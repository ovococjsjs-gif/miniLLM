#!/usr/bin/env python3
"""Choose learned-state delta strength using train prompts only."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from evaluate_qwen35_learned_state_replay import read_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alphas", default="0.01,0.025,0.05,0.075,0.1,0.25,0.5,0.75,1.0"
    )
    parser.add_argument(
        "--checkpoint", default="artifacts/qwen35-gated-delta-updater-v1/model.pt"
    )
    parser.add_argument(
        "--binary",
        default=".cache/qwen35-transition-probe/qwen35-learned-state-replay",
    )
    parser.add_argument(
        "--model",
        default="data/external/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M-github.gguf",
    )
    parser.add_argument(
        "--experiment-config",
        default="configs/experiments/qwen35_real_state_pairs_v1.json",
    )
    parser.add_argument("--work-dir", default=".cache/qwen35-alpha-cal")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument(
        "--output", default="results/qwen35_learned_state_alpha_calibration_v1.json"
    )
    args = parser.parse_args()

    alphas = [float(value) for value in args.alphas.split(",")]
    if not alphas or any(not 0 < value <= 1 for value in alphas):
        raise ValueError("all alpha candidates must lie in (0, 1]")
    experiment = json.loads(Path(args.experiment_config).read_text(encoding="utf-8"))
    prompts = [item for item in experiment["prompts"] if item["split"] == "train"]
    work = Path(args.work_dir)
    records = []
    for alpha in alphas:
        prompt_metrics: list[dict[str, Any]] = []
        for prompt in prompts:
            root = work / str(alpha) / prompt["id"]
            replay_dir = root / "replay"
            metrics_path = replay_dir / "metrics.tsv"
            if not args.reuse_existing or not metrics_path.exists():
                shutil.rmtree(root, ignore_errors=True)
                root.mkdir(parents=True)
                patch = root / "patch.bin"
                export = subprocess.run(
                    [
                        sys.executable,
                        "scripts/export_qwen35_learned_state_patch.py",
                        "--checkpoint",
                        args.checkpoint,
                        "--prompt-id",
                        prompt["id"],
                        "--stage",
                        "1",
                        "--alpha",
                        str(alpha),
                        "--output",
                        str(patch),
                    ],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if export.returncode:
                    raise RuntimeError(export.stdout + export.stderr)
                replay = subprocess.run(
                    [
                        args.binary,
                        args.model,
                        str(replay_dir),
                        prompt["text"],
                        str(patch),
                    ],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if replay.returncode:
                    raise RuntimeError(replay.stdout + replay.stderr)
            metrics = read_metrics(metrics_path)
            prompt_metrics.append(
                {
                    "prompt_id": prompt["id"],
                    "candidate_copy_kl": metrics["candidate_copy_kl"],
                    "learned_kl": metrics["learned_kl"],
                    "improved": metrics["learned_kl"] < metrics["candidate_copy_kl"],
                }
            )
        mean_copy = sum(item["candidate_copy_kl"] for item in prompt_metrics) / len(
            prompt_metrics
        )
        mean_learned = sum(item["learned_kl"] for item in prompt_metrics) / len(
            prompt_metrics
        )
        records.append(
            {
                "alpha": alpha,
                "prompts": len(prompt_metrics),
                "mean_candidate_copy_kl": mean_copy,
                "mean_learned_kl": mean_learned,
                "learned_over_copy_ratio": mean_learned / mean_copy,
                "improved_prompts": sum(item["improved"] for item in prompt_metrics),
                "records": prompt_metrics,
            }
        )
    selected = min(records, key=lambda item: item["mean_learned_kl"])
    report = {
        "schema_version": 1,
        "experiment": "qwen35-learned-state-alpha-calibration-v1",
        "split": "train-only",
        "validation_prompts_used": 0,
        "candidates": records,
        "selected_alpha": selected["alpha"],
        "selection_metric": "minimum mean train true-vocabulary KL",
        "selected_mean_kl": selected["mean_learned_kl"],
        "selected_copy_mean_kl": selected["mean_candidate_copy_kl"],
        "selected_ratio": selected["learned_over_copy_ratio"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
