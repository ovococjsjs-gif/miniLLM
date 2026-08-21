#!/usr/bin/env python3
"""One-command, resumable orchestration for the active AIra-Qwen pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from minillm.aira.transition_data import TransitionCorpus, sha256


def run(command: list[str], *, timeout: int = 1800) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            + completed.stdout
            + completed.stderr
        )
    return {
        "command": command,
        "seconds": time.perf_counter() - started,
        "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
    }


def build_if_needed(
    *,
    source: Path,
    binary: Path,
    report: Path,
    build_script: str,
) -> dict[str, Any]:
    if report.exists() and binary.exists():
        metadata = json.loads(report.read_text(encoding="utf-8"))
        if metadata.get("source_sha256") == sha256(source) and metadata.get(
            "binary_sha256"
        ) == sha256(binary):
            return {"status": "verified", "report": str(report)}
    result = run(
        [
            sys.executable,
            build_script,
            "--source",
            str(source),
            "--binary",
            str(binary),
            "--output",
            str(report),
        ]
    )
    return {"status": "built", "report": str(report), **result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/experiments/aira_qwen_active_v1.json"
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Verify/build/normalize everything but do not optimize parameters.",
    )
    parser.add_argument("--force-recollect", action="store_true")
    parser.add_argument(
        "--reuse-checkpoints",
        action="store_true",
        help="Skip optimization and package the existing hash-bound component checkpoints.",
    )
    parser.add_argument(
        "--reuse-calibration-cache",
        action="store_true",
        help="Reuse existing train-only native replay files for calibration reporting.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    scorecard_path = Path(config["scorecard"])
    scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
    if (
        config["evaluation"]["transitions"]
        != scorecard["data"]["validation_transitions"]
        or config["evaluation"]["state_delta_alpha"]
        != scorecard["intervention"]["state_delta_alpha"]
    ):
        raise ValueError("active evaluation differs from the fixed scorecard")
    donor_path = Path(config["donor_config"])
    donor = json.loads(donor_path.read_text(encoding="utf-8"))
    model_path = Path(config["model_path"])
    if model_path.stat().st_size != donor["github_mirror"]["size_bytes"]:
        raise ValueError("active donor size differs from pinned config")
    if sha256(model_path) != donor["github_mirror"]["sha256"]:
        raise ValueError("active donor hash differs from pinned config")
    llama_source = Path(config["llama_source"])
    revision = subprocess.run(
        ["git", "-C", str(llama_source), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if revision != donor["runtime"]["revision"]:
        raise ValueError("llama.cpp revision differs from active donor config")

    steps: list[dict[str, Any]] = []
    transition_build = Path("results/qwen35_transition_probe_build.json")
    replay_build = Path("results/qwen35_learned_state_replay_build.json")
    steps.append(
        {
            "transition_probe": build_if_needed(
                source=Path(config["transition_probe_source"]),
                binary=Path(config["transition_probe_binary"]),
                report=transition_build,
                build_script="scripts/build_qwen35_state_probe.py",
            )
        }
    )
    steps.append(
        {
            "learned_replay": build_if_needed(
                source=Path(config["learned_replay_source"]),
                binary=Path(config["learned_replay_binary"]),
                report=replay_build,
                build_script="scripts/build_qwen35_state_replay.py",
            )
        }
    )

    raw_root = Path(config["raw_state_work_dir"])
    needs_collection = args.force_recollect
    try:
        corpus = TransitionCorpus.load(config["source_pairs"], raw_root)
        inventory = corpus.validate_candidate_inventory(
            config["normalization"]["candidate_layers"]
        )
    except (FileNotFoundError, ValueError):
        needs_collection = True
    if needs_collection:
        collection = run(
            [
                sys.executable,
                "scripts/collect_qwen35_state_pairs.py",
                "--binary",
                config["transition_probe_binary"],
                "--build-report",
                str(transition_build),
                "--work-dir",
                str(raw_root),
                "--output-dir",
                ".cache/aira-qwen-active-projected",
                "--keep-raw",
                "--overwrite",
            ]
        )
        steps.append({"collection": collection})
        corpus = TransitionCorpus.load(config["source_pairs"], raw_root)
        inventory = corpus.validate_candidate_inventory(
            config["normalization"]["candidate_layers"]
        )
    normalization = corpus.event_normalization(
        config["normalization"]["candidate_layers"],
        minimum_standard_deviation=config["normalization"][
            "minimum_standard_deviation"
        ],
    )
    prepared = {
        "scorecard": str(scorecard_path),
        "scorecard_sha256": sha256(scorecard_path),
        "model_sha256": sha256(model_path),
        "llama_cpp_revision": revision,
        "source_manifest_sha256": sha256(corpus.manifest_path),
        "inventory": inventory,
        "normalization_records": normalization.records,
        "normalization_mean_sha256": hashlib.sha256(
            normalization.mean.numpy().tobytes()
        ).hexdigest(),
        "normalization_scale_sha256": hashlib.sha256(
            normalization.scale.numpy().tobytes()
        ).hexdigest(),
        "train_transitions": len(corpus.train_indices),
        "validation_transitions": len(corpus.validation_indices),
        "stored_answer_routes_used": False,
        "ready_for_training": True,
    }
    if args.prepare_only:
        report = {
            "schema_version": 1,
            "experiment": config["name"],
            "mode": "prepare-only",
            "config_sha256": sha256(config_path),
            "prepared": prepared,
            "steps": steps,
        }
        output = Path(config["status_report"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    if not args.reuse_checkpoints:
        steps.append(
            {
                "state_training": run(
                    [sys.executable, "scripts/train_qwen35_gated_delta_updater.py"]
                )
            }
        )
        steps.append(
            {
                "conv_training": run(
                    [sys.executable, "scripts/train_qwen35_conv_row_updater.py"]
                )
            }
        )
    steps.append(
        {
            "combined_checkpoint": run(
                [
                    sys.executable,
                    "scripts/package_aira_qwen_cache_updater.py",
                    "--config",
                    str(config_path),
                ]
            )
        }
    )
    combined_checkpoint = Path(config["combined_checkpoint"])
    training_report = {
        "state": json.loads(
            Path("results/qwen35_gated_delta_updater_v1.json").read_text(
                encoding="utf-8"
            )
        ),
        "convolution": json.loads(
            Path("results/qwen35_conv_row_updater_v1.json").read_text(encoding="utf-8")
        ),
        "combined": json.loads(
            Path("results/aira_qwen_combined_checkpoint_v1.json").read_text(
                encoding="utf-8"
            )
        ),
    }

    calibration_path = Path(config["status_report"]).with_name(
        "aira_qwen_active_alpha_calibration_v1.json"
    )
    alpha_values = ",".join(
        str(value) for value in config["calibration"]["state_delta_alpha_candidates"]
    )
    calibration_command = [
        sys.executable,
        "scripts/calibrate_qwen35_learned_state_alpha.py",
        "--combined-checkpoint",
        str(combined_checkpoint),
        "--metric",
        "full-cache",
        "--alphas",
        alpha_values,
        "--work-dir",
        ".cache/aira-qwen-active-alpha",
        "--output",
        str(calibration_path),
    ]
    if args.reuse_calibration_cache:
        calibration_command.append("--reuse-existing")
    steps.append({"calibration": run(calibration_command)})
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    replay_path = Path(config["status_report"]).with_name(
        "aira_qwen_active_replay_v1.json"
    )
    steps.append(
        {
            "replay": run(
                [
                    sys.executable,
                    "scripts/evaluate_qwen35_learned_state_replay.py",
                    "--combined-checkpoint",
                    str(combined_checkpoint),
                    "--alpha",
                    str(config["evaluation"]["state_delta_alpha"]),
                    "--work-dir",
                    ".cache/aira-qwen-active-replay",
                    "--output",
                    str(replay_path),
                ]
            )
        }
    )
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    actual_scorecard = {
        "full_state_mse_ratio": training_report["state"]["validation"][
            "state_mse_ratio"
        ],
        "conv_new_row_mse_ratio": training_report["convolution"]["validation"][
            "new_row_mse_ratio"
        ],
        "full_cache_true_vocabulary_kl_ratio": replay[
            "learned_full_over_copy_kl_ratio"
        ],
        "full_cache_improved_transitions": replay["learned_full_improvements"],
        "full_cache_oracle_argmax_preserved": replay["learned_full_argmax_preserved"],
    }
    scorecard_result = {}
    for name, specification in scorecard["headline_metrics"].items():
        actual = actual_scorecard[name]
        limit = specification["regression_limit"]
        passed = (
            actual <= limit
            if specification["direction"] == "lower"
            else actual >= limit
        )
        scorecard_result[name] = {
            "actual": actual,
            "reference": specification["reference"],
            "regression_limit": limit,
            "passed": passed,
        }
    report = {
        "schema_version": 1,
        "experiment": config["name"],
        "mode": "trained-and-replayed",
        "config_sha256": sha256(config_path),
        "prepared": prepared,
        "combined_checkpoint": str(combined_checkpoint),
        "combined_checkpoint_sha256": sha256(combined_checkpoint),
        "training": training_report,
        "calibration": calibration,
        "applied_intervention": {
            "state_delta_alpha": config["evaluation"]["state_delta_alpha"],
            "policy": config["evaluation"]["alpha_policy"],
            "diagnostic_full_cache_train_optimum": calibration["selected_alpha"],
        },
        "fixed_scorecard": {
            "path": str(scorecard_path),
            "sha256": sha256(scorecard_path),
            "metrics": scorecard_result,
            "all_passed": all(item["passed"] for item in scorecard_result.values()),
        },
        "replay": {key: value for key, value in replay.items() if key != "records"},
        "gates": {
            "normalization_contract_hash_bound": True,
            "fixed_scorecard_all_passed": all(
                item["passed"] for item in scorecard_result.values()
            ),
            "combined_checkpoint": True,
            "full_cache_mean_kl": replay["acceptance"][
                "learned_full_mean_kl_below_full_copy"
            ],
            "every_transition_kl": replay["acceptance"][
                "learned_full_improves_every_prompt"
            ],
            "autoregressive_generation": False,
            "physical_skip_speedup": False,
            "deployment_allowed": False,
        },
        "steps": steps,
    }
    output = Path(config["status_report"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
