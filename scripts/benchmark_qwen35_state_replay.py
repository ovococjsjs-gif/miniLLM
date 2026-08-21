#!/usr/bin/env python3
"""Measure true-vocabulary sensitivity to stale/interpolated Qwen recurrent states."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import subprocess
from itertools import pairwise
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_metrics(path: Path) -> dict[str, str]:
    output = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split("\t", 1)
        output[key] = value
    return output


def numeric(metrics: dict[str, str], name: str) -> float:
    return float(metrics[name])


def strategy_metrics(metrics: dict[str, str], prefix: str) -> dict[str, Any]:
    return {
        "kl": numeric(metrics, f"{prefix}_kl"),
        "rms": numeric(metrics, f"{prefix}_rms"),
        "max_abs": numeric(metrics, f"{prefix}_max_abs"),
        "argmax": int(metrics[f"{prefix}_argmax"]),
        "argmax_matches_oracle": (
            int(metrics[f"{prefix}_argmax"]) == int(metrics["oracle_argmax"])
        ),
    }


def aggregate(records: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
    items = [record["strategies"][strategy] for record in records]
    kls = [item["kl"] for item in items]
    rms = [item["rms"] for item in items]
    return {
        "records": len(items),
        "mean_kl": statistics.fmean(kls),
        "median_kl": statistics.median(kls),
        "max_kl": max(kls),
        "mean_logit_rms": statistics.fmean(rms),
        "argmax_matches": sum(item["argmax_matches_oracle"] for item in items),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-config",
        default="configs/experiments/qwen35_real_state_pairs_v1.json",
    )
    parser.add_argument("--donor-config", default="configs/donors/qwen35_08b.json")
    parser.add_argument(
        "--model",
        default=(
            "data/external/qwen3.5-0.8b/"
            "Qwen3.5-0.8B-Q4_K_M-github.gguf"
        ),
    )
    parser.add_argument(
        "--binary", default=".cache/qwen35-state-probe/qwen35-state-replay"
    )
    parser.add_argument(
        "--build-report", default="results/qwen35_state_replay_build.json"
    )
    parser.add_argument("--work-dir", default="data/qwen35-state-replay-work")
    parser.add_argument(
        "--output", default="results/qwen35_state_replay_baseline.json"
    )
    args = parser.parse_args()

    experiment_path = Path(args.experiment_config)
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    donor = json.loads(Path(args.donor_config).read_text(encoding="utf-8"))
    build_report_path = Path(args.build_report)
    build_report = json.loads(build_report_path.read_text(encoding="utf-8"))
    model = Path(args.model)
    binary = Path(args.binary)
    if sha256(model) != donor["github_mirror"]["sha256"]:
        raise ValueError("replay model differs from pinned donor")
    if sha256(binary) != build_report["binary_sha256"]:
        raise ValueError("replay binary differs from build report")

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    records = []
    for prompt in experiment["prompts"]:
        raw = work / prompt["id"]
        shutil.rmtree(raw, ignore_errors=True)
        completed = subprocess.run(
            [str(binary), str(model), str(raw), prompt["text"]],
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode:
            raw.mkdir(parents=True, exist_ok=True)
            (raw / "stdout.log").write_text(completed.stdout, encoding="utf-8")
            (raw / "stderr.log").write_text(completed.stderr, encoding="utf-8")
            raise RuntimeError(f"state replay failed for {prompt['id']}")
        metrics = read_metrics(raw / "metrics.tsv")
        control = {
            "exact": metrics["control_exact"] == "1",
            "kl": numeric(metrics, "control_kl"),
            "rms": numeric(metrics, "control_rms"),
            "max_abs": numeric(metrics, "control_max_abs"),
        }
        if not control["exact"] or any(control[key] != 0 for key in ("kl", "rms", "max_abs")):
            raise ValueError(f"serialization control is not exact for {prompt['id']}")
        strategies = {
            "stale_0": strategy_metrics(metrics, "stale"),
            "oracle_blend_25": strategy_metrics(metrics, "blend_25"),
            "oracle_blend_50": strategy_metrics(metrics, "blend_50"),
            "oracle_blend_75": strategy_metrics(metrics, "blend_75"),
        }
        logits_hashes = {
            name: sha256(path)
            for name, path in {
                "oracle": raw / "oracle.logits.f32.bin",
                "control": raw / "control.logits.f32.bin",
                "stale_0": raw / "stale.logits.f32.bin",
                "oracle_blend_25": raw / "blend-25.logits.f32.bin",
                "oracle_blend_50": raw / "blend-50.logits.f32.bin",
                "oracle_blend_75": raw / "blend-75.logits.f32.bin",
            }.items()
        }
        records.append(
            {
                "prompt_id": prompt["id"],
                "split": prompt["split"],
                "prompt_sha256": hashlib.sha256(prompt["text"].encode()).hexdigest(),
                "prompt_tokens": int(metrics["prompt_tokens"]),
                "first_token": int(metrics["first_token"]),
                "first_piece_hex": metrics["first_piece_hex"],
                "second_token": int(metrics["second_token"]),
                "second_piece_hex": metrics["second_piece_hex"],
                "oracle_argmax": int(metrics["oracle_argmax"]),
                "full_after_bytes": int(metrics["full_after_bytes"]),
                "partial_state_bytes": int(metrics["partial_after_bytes"]),
                "partial_data_offset": int(metrics["partial_after_data_offset"]),
                "serialization_control": control,
                "strategies": strategies,
                "logit_sha256": logits_hashes,
            }
        )
        shutil.rmtree(raw)

    strategies = ("stale_0", "oracle_blend_25", "oracle_blend_50", "oracle_blend_75")
    aggregates = {strategy: aggregate(records, strategy) for strategy in strategies}
    monotonic = 0
    for record in records:
        curve = [record["strategies"][strategy]["kl"] for strategy in strategies]
        if all(left >= right for left, right in pairwise(curve)):
            monotonic += 1
    report = {
        "schema_version": 1,
        "experiment": "qwen35-true-vocabulary-recurrent-state-replay-v1",
        "role": "stale-state baseline and oracle interpolation sensitivity; not a deployable patcher",
        "model_sha256": sha256(model),
        "llama_cpp_revision": donor["runtime"]["revision"],
        "replay_source_sha256": build_report["source_sha256"],
        "replay_binary_sha256": build_report["binary_sha256"],
        "experiment_config": str(experiment_path),
        "experiment_config_sha256": sha256(experiment_path),
        "records": len(records),
        "serialization_controls_exact": sum(
            record["serialization_control"]["exact"] for record in records
        ),
        "monotonic_kl_curves": monotonic,
        "aggregates": aggregates,
        "samples": records,
        "interpretation": {
            "attention_kv_preserved_from_oracle_event": True,
            "only_recurrent_and_convolution_state_replaced": True,
            "stale_state_is_safe": aggregates["stale_0"]["mean_kl"] < 0.01,
            "oracle_interpolation_is_deployable": False,
            "learned_full_state_update_tested": False,
            "acceleration_claim_allowed": False,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "records": len(records),
                "serialization_controls_exact": report[
                    "serialization_controls_exact"
                ],
                "monotonic_kl_curves": monotonic,
                "aggregates": aggregates,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
