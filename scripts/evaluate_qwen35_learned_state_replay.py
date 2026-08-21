#!/usr/bin/env python3
"""Evaluate learned full-state patches through public Qwen state injection."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_metrics(path: Path) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split("\t")
        try:
            output[key] = (
                float(value) if any(marker in value for marker in ".eE") else int(value)
            )
        except ValueError:
            output[key] = value
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="data/external/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M-github.gguf",
    )
    parser.add_argument("--combined-checkpoint")
    parser.add_argument(
        "--checkpoint", default="artifacts/qwen35-gated-delta-updater-v1/model.pt"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Train-calibrated fraction of the learned state delta to inject.",
    )
    parser.add_argument("--stages", default="1,2,3,4")
    parser.add_argument(
        "--binary",
        default=".cache/qwen35-transition-probe/qwen35-learned-state-replay",
    )
    parser.add_argument(
        "--binary-build", default="results/qwen35_learned_state_replay_build.json"
    )
    parser.add_argument(
        "--experiment-config",
        default="configs/experiments/qwen35_real_state_pairs_v1.json",
    )
    parser.add_argument("--work-dir", default=".cache/qwen35-learned-replay-eval")
    parser.add_argument(
        "--output", default="results/qwen35_learned_state_replay_v1.json"
    )
    args = parser.parse_args()

    build = json.loads(Path(args.binary_build).read_text(encoding="utf-8"))
    if sha256(Path(args.binary)) != build["binary_sha256"]:
        raise ValueError("learned replay binary differs from build provenance")
    experiment = json.loads(Path(args.experiment_config).read_text(encoding="utf-8"))
    prompts = [item for item in experiment["prompts"] if item["split"] == "validation"]
    stages = [int(value) for value in args.stages.split(",")]
    if not stages or any(
        stage < 1 or stage > experiment["continuation_tokens"] for stage in stages
    ):
        raise ValueError("replay stages are outside the captured continuation")
    work = Path(args.work_dir)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    records = []
    started = time.perf_counter()
    for prompt in prompts:
        for stage in stages:
            root = work / prompt["id"] / f"stage-{stage}"
            root.mkdir(parents=True)
            patch = root / "patch.bin"
            export_command = [
                sys.executable,
                "scripts/export_qwen35_learned_state_patch.py",
                "--checkpoint",
                args.checkpoint,
                "--prompt-id",
                prompt["id"],
                "--stage",
                str(stage),
                "--alpha",
                str(args.alpha),
                "--output",
                str(patch),
            ]
            if args.combined_checkpoint:
                export_command.extend(
                    ["--combined-checkpoint", args.combined_checkpoint]
                )
            export = subprocess.run(
                export_command,
                text=True,
                capture_output=True,
                check=False,
            )
            if export.returncode:
                raise RuntimeError(export.stdout + export.stderr)
            replay_dir = root / "replay"
            replay = subprocess.run(
                [args.binary, args.model, str(replay_dir), prompt["text"], str(patch)],
                text=True,
                capture_output=True,
                check=False,
            )
            if replay.returncode:
                raise RuntimeError(replay.stdout + replay.stderr)
            patch_metadata = json.loads(
                patch.with_suffix(".json").read_text(encoding="utf-8")
            )
            metrics = read_metrics(replay_dir / "metrics.tsv")
            records.append(
                {
                    "prompt_id": prompt["id"],
                    "stage": stage,
                    "prompt_sha256": hashlib.sha256(
                        prompt["text"].encode()
                    ).hexdigest(),
                    "state_mse_ratio": patch_metadata["state_mse_ratio"],
                    "conv_new_row_mse_ratio": patch_metadata["conv_new_row_mse_ratio"],
                    "candidate_copy_kl": metrics["candidate_copy_kl"],
                    "learned_kl": metrics["learned_kl"],
                    "learned_kl_improvement": metrics["learned_kl_improvement"],
                    "oracle_argmax": metrics["oracle_argmax"],
                    "candidate_copy_argmax": metrics["candidate_copy_argmax"],
                    "learned_argmax": metrics["learned_argmax"],
                    "candidate_full_copy_kl": metrics["candidate_full_copy_kl"],
                    "learned_full_kl": metrics["learned_full_kl"],
                    "learned_full_kl_improvement": metrics[
                        "learned_full_kl_improvement"
                    ],
                    "candidate_full_copy_argmax": metrics["candidate_full_copy_argmax"],
                    "learned_full_argmax": metrics["learned_full_argmax"],
                    "serialization_control_exact": bool(metrics["control_exact"]),
                }
            )
    mean_copy = sum(item["candidate_copy_kl"] for item in records) / len(records)
    mean_learned = sum(item["learned_kl"] for item in records) / len(records)
    mean_full_copy = sum(item["candidate_full_copy_kl"] for item in records) / len(
        records
    )
    mean_learned_full = sum(item["learned_full_kl"] for item in records) / len(records)
    report = {
        "schema_version": 1,
        "experiment": "qwen35-learned-state-replay-v1",
        "role": "native injection gate for learned full recurrent states",
        "checkpoint_sha256": sha256(Path(args.combined_checkpoint or args.checkpoint)),
        "combined_checkpoint": args.combined_checkpoint,
        "binary_build": args.binary_build,
        "binary_sha256": build["binary_sha256"],
        "prompt_groups": len(prompts),
        "stages": stages,
        "transitions": len(records),
        "train_calibrated_alpha": args.alpha,
        "alpha_calibration_scope": "train prompts, stage 1 only",
        "mean_candidate_copy_kl": mean_copy,
        "mean_learned_kl": mean_learned,
        "learned_over_copy_kl_ratio": mean_learned / mean_copy,
        "learned_improvements": sum(
            item["learned_kl_improvement"] > 0 for item in records
        ),
        "argmax_preserved": sum(
            item["learned_argmax"] == item["oracle_argmax"] for item in records
        ),
        "mean_candidate_full_copy_kl": mean_full_copy,
        "mean_learned_full_kl": mean_learned_full,
        "learned_full_over_copy_kl_ratio": mean_learned_full / mean_full_copy,
        "learned_full_improvements": sum(
            item["learned_full_kl_improvement"] > 0 for item in records
        ),
        "learned_full_argmax_preserved": sum(
            item["learned_full_argmax"] == item["oracle_argmax"] for item in records
        ),
        "records": records,
        "elapsed_seconds": time.perf_counter() - started,
        "acceptance": {
            "all_serialization_controls_exact": all(
                item["serialization_control_exact"] for item in records
            ),
            "learned_mean_kl_below_candidate_copy": mean_learned < mean_copy,
            "learned_improves_every_prompt": all(
                item["learned_kl_improvement"] > 0 for item in records
            ),
            "oracle_convolution_removed": True,
            "learned_full_mean_kl_below_full_copy": mean_learned_full < mean_full_copy,
            "learned_full_improves_every_prompt": all(
                item["learned_full_kl_improvement"] > 0 for item in records
            ),
            "generated_quality_gate_passed": False,
            "speedup_gate_passed": False,
            "deployment_allowed": False,
        },
        "interpretation": (
            "Injection is the binding test. State alpha is selected only on train "
            "prompts. The strict full baseline leaves both candidate recurrent and "
            "convolution states stale; learned full patches replace both without oracle "
            "convolution. Generated-quality and speed gates remain separate."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
