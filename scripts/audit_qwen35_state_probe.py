#!/usr/bin/env python3
"""Validate exact Qwen3.5 recurrent-state captures and write a compact audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from array import array
from collections import Counter
from pathlib import Path
from typing import Any

_RECURRENT_LAYERS = (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22)
_LAYER_NAME = re.compile(r"^(?P<base>[a-z_]+)-(?P<layer>\d+)$")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_key_values(path: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split("\t", 1)
        output[key] = value
    return output


def read_f32(path: Path) -> array[float]:
    values = array("f")
    values.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        values.byteswap()
    return values


def bundle_sha256(root: Path, rows: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["name"].encode())
        digest.update(b"\0")
        with (root / row["file"]).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def layer_rows(
    rows: list[dict[str, str]], stage: int, base: str
) -> dict[int, dict[str, str]]:
    output: dict[int, dict[str, str]] = {}
    for row in rows:
        if int(row["stage"]) != stage:
            continue
        match = _LAYER_NAME.fullmatch(row["name"])
        if match and match.group("base") == base:
            output[int(match.group("layer"))] = row
    return output


def vector_change(before: array[float], after: array[float]) -> dict[str, float]:
    if len(before) != len(after):
        raise ValueError("state vectors have different lengths")
    before_sq = 0.0
    after_sq = 0.0
    delta_sq = 0.0
    dot = 0.0
    for left, right in zip(before, after, strict=True):
        before_sq += left * left
        after_sq += right * right
        difference = right - left
        delta_sq += difference * difference
        dot += left * right
    before_l2 = math.sqrt(before_sq)
    after_l2 = math.sqrt(after_sq)
    delta_l2 = math.sqrt(delta_sq)
    return {
        "before_l2": before_l2,
        "after_l2": after_l2,
        "delta_l2": delta_l2,
        "delta_over_after": delta_l2 / max(after_l2, 1e-30),
        "cosine": dot / max(before_l2 * after_l2, 1e-30),
    }


def audit(
    raw_dir: Path,
    *,
    model_path: Path,
    config_path: Path,
    build_report_path: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    build_report = json.loads(build_report_path.read_text(encoding="utf-8"))
    run = read_key_values(raw_dir / "run.tsv")
    continuation_tokens = int(run["continuation_tokens"])
    stage_count = continuation_tokens + 1
    stages = tuple(range(stage_count))

    rows = list(
        csv.DictReader((raw_dir / "captures.tsv").open(encoding="utf-8"), delimiter="\t")
    )
    if not rows:
        raise ValueError("probe contains no tensor captures")
    for row in rows:
        capture = raw_dir / row["file"]
        if not capture.is_file():
            raise FileNotFoundError(capture)
        if capture.stat().st_size != int(row["bytes"]):
            raise ValueError(f"capture size mismatch: {capture}")
        if row["type"] != "f32":
            raise ValueError(f"unexpected capture type: {row['type']}")
        if int(row["bytes"]) != int(row["elements"]) * 4:
            raise ValueError(f"capture is not packed float32: {capture}")

    expected_recurrent = set(_RECURRENT_LAYERS)
    captures_per_stage: dict[str, int] = {}
    state_shapes: set[str] = set()
    conv_shapes: dict[str, set[str]] = {"before": set(), "after": set()}
    hidden_shapes: dict[str, Counter[str]] = {}
    stage_bundles: dict[str, dict[str, str]] = {}
    state_dynamics: dict[str, dict[str, Any]] = {}

    for stage in stages:
        stage_rows = [row for row in rows if int(row["stage"]) == stage]
        captures_per_stage[str(stage)] = len(stage_rows)
        before = layer_rows(rows, stage, "state_predelta")
        after = layer_rows(rows, stage, "new_state")
        conv_before = layer_rows(rows, stage, "conv_states")
        conv_after = layer_rows(rows, stage, "last_conv_states")
        if set(before) != expected_recurrent or set(after) != expected_recurrent:
            raise ValueError(f"stage {stage} recurrent layer inventory is incomplete")
        if set(conv_before) != expected_recurrent or set(conv_after) != expected_recurrent:
            raise ValueError(f"stage {stage} convolution-state inventory is incomplete")
        state_shapes.update(row["shape"] for row in before.values())
        state_shapes.update(row["shape"] for row in after.values())
        conv_shapes["before"].update(row["shape"] for row in conv_before.values())
        conv_shapes["after"].update(row["shape"] for row in conv_after.values())

        hidden = [
            row
            for row in stage_rows
            if row["name"] == "h_pre_norm" or row["name"].startswith("l_out-")
        ]
        if len(hidden) != 24:
            raise ValueError(f"stage {stage} does not contain 24 layer outputs")
        hidden_shapes[str(stage)] = Counter(row["shape"] for row in hidden)

        stage_bundles[str(stage)] = {
            "state_before_sha256": bundle_sha256(
                raw_dir, [before[layer] for layer in _RECURRENT_LAYERS]
            ),
            "state_after_sha256": bundle_sha256(
                raw_dir, [after[layer] for layer in _RECURRENT_LAYERS]
            ),
            "conv_before_sha256": bundle_sha256(
                raw_dir, [conv_before[layer] for layer in _RECURRENT_LAYERS]
            ),
            "conv_after_sha256": bundle_sha256(
                raw_dir, [conv_after[layer] for layer in _RECURRENT_LAYERS]
            ),
            "hidden_sha256": bundle_sha256(raw_dir, hidden),
        }

        layer_changes: dict[str, Any] = {}
        for layer in _RECURRENT_LAYERS:
            layer_changes[str(layer)] = vector_change(
                read_f32(raw_dir / before[layer]["file"]),
                read_f32(raw_dir / after[layer]["file"]),
            )
        ratios = [item["delta_over_after"] for item in layer_changes.values()]
        state_dynamics[str(stage)] = {
            "layers": layer_changes,
            "mean_delta_over_after": sum(ratios) / len(ratios),
            "max_delta_over_after": max(ratios),
        }

    state_chain_matches = 0
    conv_chain_matches = 0
    chain_comparisons = max(0, stage_count - 1) * len(_RECURRENT_LAYERS)
    for stage in range(stage_count - 1):
        current_state = layer_rows(rows, stage, "new_state")
        next_state = layer_rows(rows, stage + 1, "state_predelta")
        current_conv = layer_rows(rows, stage, "last_conv_states")
        next_conv = layer_rows(rows, stage + 1, "conv_states")
        for layer in _RECURRENT_LAYERS:
            if sha256(raw_dir / current_state[layer]["file"]) == sha256(
                raw_dir / next_state[layer]["file"]
            ):
                state_chain_matches += 1
            if sha256(raw_dir / current_conv[layer]["file"]) == sha256(
                raw_dir / next_conv[layer]["file"]
            ):
                conv_chain_matches += 1

    token_rows = list(
        csv.DictReader((raw_dir / "tokens.tsv").open(encoding="utf-8"), delimiter="\t")
    )
    if len(token_rows) != stage_count:
        raise ValueError("logit stage count does not match state stage count")
    vocabulary = int(run["vocabulary"])
    logits = []
    for token_row in token_rows:
        logits_path = raw_dir / token_row["logits_file"]
        if logits_path.stat().st_size != vocabulary * 4:
            raise ValueError(f"invalid full-logit capture: {logits_path}")
        top_items = [token_row["top10_token_logit"], *(token_row.get(None) or [])]
        logits.append(
            {
                "stage": int(token_row["stage"]),
                "selected_token": int(token_row["selected_token"]),
                "selected_piece_hex": token_row["selected_piece_hex"],
                "full_vector_sha256": sha256(logits_path),
                "elements": vocabulary,
                "top10": top_items,
            }
        )

    raw_files = [path for path in raw_dir.iterdir() if path.is_file()]
    model_hash = sha256(model_path)
    expected_hash = config["github_mirror"]["sha256"]
    if model_hash != expected_hash:
        raise ValueError("state probe model hash differs from pinned donor")
    exact_state_chain = state_chain_matches == chain_comparisons
    exact_conv_chain = conv_chain_matches == chain_comparisons
    return {
        "schema_version": 1,
        "role": "real-qwen-state-extraction-gate-not-a-patcher-quality-claim",
        "model": {
            "path": str(model_path),
            "sha256": model_hash,
            "description": run["model"],
            "layers": int(run["layers"]),
            "embedding": int(run["embedding"]),
            "vocabulary": vocabulary,
        },
        "runtime": {
            "llama_cpp_revision": config["runtime"]["revision"],
            "probe_source_sha256": build_report["source_sha256"],
            "probe_binary_sha256": build_report["binary_sha256"],
        },
        "input": {
            "prompt_sha256": sha256(raw_dir / "prompt.txt"),
            "prompt_tokens": int(run["prompt_tokens"]),
            "continuation_tokens": continuation_tokens,
            "generated_hex": run["generated_hex"],
        },
        "architecture": {
            "recurrent_layers": list(_RECURRENT_LAYERS),
            "attention_layers": [3, 7, 11, 15, 19, 23],
            "state_elements_per_recurrent_layer": 262144,
            "state_bytes_per_recurrent_layer": 1048576,
            "conv_state_elements_per_recurrent_layer": 18432,
            "state_shapes": sorted(state_shapes),
            "conv_shapes": {
                key: sorted(value) for key, value in conv_shapes.items()
            },
        },
        "captures": {
            "raw_directory": str(raw_dir),
            "stages": stage_count,
            "captures_per_stage": captures_per_stage,
            "total_tensor_captures": len(rows),
            "total_raw_files": len(raw_files),
            "total_raw_bytes": sum(path.stat().st_size for path in raw_files),
            "hidden_shapes": {
                stage: dict(sorted(counts.items()))
                for stage, counts in hidden_shapes.items()
            },
            "stage_bundles": stage_bundles,
        },
        "cache_continuity": {
            "comparisons": chain_comparisons,
            "recurrent_state_exact_matches": state_chain_matches,
            "convolution_state_exact_matches": conv_chain_matches,
            "recurrent_state_exact": exact_state_chain,
            "convolution_state_exact": exact_conv_chain,
        },
        "state_dynamics": state_dynamics,
        "future_logits": logits,
        "gate_interpretation": {
            "real_recurrent_states_captured": True,
            "full_logits_captured": True,
            "cache_chain_exact": exact_state_chain and exact_conv_chain,
            "state_patcher_trained_on_real_states": False,
            "acceleration_claim_allowed": False,
            "next_requirement": (
                "compile projected real-state pairs, train the bounded patcher, and compare "
                "future logits plus generated quality against the full donor"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/qwen35-state-probe-smoke")
    parser.add_argument(
        "--model",
        default=(
            "data/external/qwen3.5-0.8b/"
            "Qwen3.5-0.8B-Q4_K_M-github.gguf"
        ),
    )
    parser.add_argument("--config", default="configs/donors/qwen35_08b.json")
    parser.add_argument(
        "--build-report", default="results/qwen35_state_probe_build.json"
    )
    parser.add_argument(
        "--output", default="results/qwen35_08b_real_state_probe.json"
    )
    args = parser.parse_args()
    report = audit(
        Path(args.raw_dir),
        model_path=Path(args.model),
        config_path=Path(args.config),
        build_report_path=Path(args.build_report),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "stages": report["captures"]["stages"],
                "tensor_captures": report["captures"]["total_tensor_captures"],
                "cache_continuity": report["cache_continuity"],
                "state_patcher_trained_on_real_states": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
