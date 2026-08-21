#!/usr/bin/env python3
"""Collect compact projected state-patcher pairs from exact Qwen recurrent states."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

_RECURRENT_LAYERS = (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22)
_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def splitmix64(values: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        output = (values + np.uint64(0x9E3779B97F4A7C15)) & _MASK64
        output = ((output ^ (output >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _MASK64
        output = ((output ^ (output >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _MASK64
        return output ^ (output >> np.uint64(31))


def count_sketch(values: np.ndarray, buckets: int, seed: int) -> np.ndarray:
    indices = np.arange(values.size, dtype=np.uint64) + np.uint64(seed)
    hashed = splitmix64(indices)
    destinations = np.asarray(hashed % np.uint64(buckets), dtype=np.int64)
    signs = np.where((hashed >> np.uint64(63)) != 0, 1.0, -1.0)
    projected = np.bincount(
        destinations,
        weights=np.asarray(values, dtype=np.float64) * signs,
        minlength=buckets,
    )
    projected /= math.sqrt(max(1.0, values.size / buckets))
    return np.asarray(projected, dtype=np.float32)


def probability_sketch(logits: np.ndarray, buckets: int, seed: int) -> np.ndarray:
    shifted = np.asarray(logits, dtype=np.float64) - float(np.max(logits))
    probabilities = np.exp(shifted)
    probabilities /= float(np.sum(probabilities))
    indices = np.arange(logits.size, dtype=np.uint64) + np.uint64(seed)
    destinations = np.asarray(splitmix64(indices) % np.uint64(buckets), dtype=np.int64)
    projected = np.bincount(destinations, weights=probabilities, minlength=buckets)
    projected = np.asarray(projected, dtype=np.float32)
    projected /= float(projected.sum())
    return projected


def event_features(token_id: int, dimensions: int) -> np.ndarray:
    if dimensions % 2:
        raise ValueError("event feature count must be even")
    positions = np.arange(dimensions // 2, dtype=np.float64)
    frequencies = np.exp(-math.log(10000.0) * positions / max(1, dimensions // 2 - 1))
    phase = (token_id + 1) * frequencies
    return np.asarray(np.concatenate((np.sin(phase), np.cos(phase))), dtype=np.float32)


def capture_rows(raw_dir: Path) -> tuple[list[dict[str, str]], dict[tuple[int, str], dict[str, str]]]:
    rows = list(
        csv.DictReader((raw_dir / "captures.tsv").open(encoding="utf-8"), delimiter="\t")
    )
    indexed = {(int(row["stage"]), row["name"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("duplicate stage/tensor capture")
    return rows, indexed


def bundle_hash(raw_dir: Path, rows: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["name"].encode())
        digest.update(b"\0")
        digest.update((raw_dir / row["file"]).read_bytes())
    return digest.hexdigest()


def exact_chain_matches(
    raw_dir: Path,
    indexed: dict[tuple[int, str], dict[str, str]],
    stages: int,
) -> tuple[int, int]:
    states = 0
    conv = 0
    for stage in range(stages - 1):
        for layer in _RECURRENT_LAYERS:
            current = raw_dir / indexed[(stage, f"new_state-{layer}")]["file"]
            following = raw_dir / indexed[(stage + 1, f"state_predelta-{layer}")]["file"]
            states += sha256(current) == sha256(following)
            current_conv = raw_dir / indexed[(stage, f"last_conv_states-{layer}")]["file"]
            following_conv = raw_dir / indexed[(stage + 1, f"conv_states-{layer}")]["file"]
            conv += sha256(current_conv) == sha256(following_conv)
    return states, conv


def projected_state(
    raw_dir: Path,
    indexed: dict[tuple[int, str], dict[str, str]],
    *,
    stage: int,
    state_base: str,
    conv_base: str,
    state_buckets: int,
    conv_buckets: int,
    seed: int,
) -> np.ndarray:
    layers = []
    for layer in _RECURRENT_LAYERS:
        state_row = indexed[(stage, f"{state_base}-{layer}")]
        conv_row = indexed[(stage, f"{conv_base}-{layer}")]
        state = np.fromfile(raw_dir / state_row["file"], dtype="<f4")
        conv = np.fromfile(raw_dir / conv_row["file"], dtype="<f4")
        if state.size != 262144 or conv.size != 18432:
            raise ValueError("unexpected exact Qwen state dimensions")
        layer_seed = seed + layer * 1_000_003
        layers.append(
            np.concatenate(
                (
                    count_sketch(state, state_buckets, layer_seed),
                    count_sketch(conv, conv_buckets, layer_seed + 97_409),
                )
            )
        )
    return np.stack(layers).astype(np.float32)


def write_array(path: Path, value: np.ndarray) -> dict[str, Any]:
    np.save(path, value, allow_pickle=False)
    return {
        "path": path.name,
        "sha256": sha256(path),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "bytes": path.stat().st_size,
    }


def collect(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.experiment_config)
    experiment = json.loads(config_path.read_text(encoding="utf-8"))
    donor_config_path = Path(args.donor_config)
    donor = json.loads(donor_config_path.read_text(encoding="utf-8"))
    build_report_path = Path(args.build_report)
    build_report = json.loads(build_report_path.read_text(encoding="utf-8"))
    model = Path(args.model)
    binary = Path(args.binary)
    if sha256(model) != donor["github_mirror"]["sha256"]:
        raise ValueError("state-pair donor differs from pinned model")
    if sha256(binary) != build_report["binary_sha256"]:
        raise ValueError("state-pair probe differs from build report")

    output = Path(args.output_dir)
    temporary_output = output.with_name(output.name + ".tmp")
    work = Path(args.work_dir)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {output}")
    shutil.rmtree(temporary_output, ignore_errors=True)
    temporary_output.mkdir(parents=True)
    work.mkdir(parents=True, exist_ok=True)

    before_samples = []
    after_samples = []
    event_samples = []
    future_samples = []
    byte_values: list[bytes] = []
    split_values = []
    token_ids = []
    metadata = []
    state_chain_matches = 0
    conv_chain_matches = 0
    chain_comparisons = 0
    continuation = int(experiment["continuation_tokens"])
    stages = continuation + 1
    seed = int(experiment["seed"])

    for prompt_index, prompt in enumerate(experiment["prompts"]):
        raw_dir = work / prompt["id"]
        shutil.rmtree(raw_dir, ignore_errors=True)
        command = [
            str(binary),
            str(model),
            str(raw_dir),
            prompt["text"],
            str(continuation),
        ]
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
        if completed.returncode:
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / "probe.stdout.log").write_text(completed.stdout, encoding="utf-8")
            (raw_dir / "probe.stderr.log").write_text(completed.stderr, encoding="utf-8")
            raise RuntimeError(f"probe failed for {prompt['id']}; raw logs preserved")

        _, indexed = capture_rows(raw_dir)
        prompt_state_matches, prompt_conv_matches = exact_chain_matches(
            raw_dir, indexed, stages
        )
        expected = continuation * len(_RECURRENT_LAYERS)
        if prompt_state_matches != expected or prompt_conv_matches != expected:
            raise ValueError(f"non-exact cache chain for {prompt['id']}")
        state_chain_matches += prompt_state_matches
        conv_chain_matches += prompt_conv_matches
        chain_comparisons += expected

        tokens = list(
            csv.DictReader((raw_dir / "tokens.tsv").open(encoding="utf-8"), delimiter="\t")
        )
        if len(tokens) != stages:
            raise ValueError(f"logit stage mismatch for {prompt['id']}")
        for stage in range(1, stages):
            before = projected_state(
                raw_dir,
                indexed,
                stage=stage,
                state_base="state_predelta",
                conv_base="conv_states",
                state_buckets=int(experiment["state_buckets"]),
                conv_buckets=int(experiment["conv_buckets"]),
                seed=seed,
            )
            after = projected_state(
                raw_dir,
                indexed,
                stage=stage,
                state_base="new_state",
                conv_base="last_conv_states",
                state_buckets=int(experiment["state_buckets"]),
                conv_buckets=int(experiment["conv_buckets"]),
                seed=seed,
            )
            consumed = tokens[stage - 1]
            current = tokens[stage]
            token_id = int(consumed["selected_token"])
            piece = bytes.fromhex(consumed["selected_piece_hex"])
            logits_path = raw_dir / current["logits_file"]
            logits = np.fromfile(logits_path, dtype="<f4")
            if logits.size != int(donor["upstream"]["vocabulary_size"]):
                raise ValueError("future logit vector has the wrong vocabulary")

            before_samples.append(before)
            after_samples.append(after)
            event_samples.append(
                event_features(token_id, int(experiment["event_features"]))
            )
            future_samples.append(
                probability_sketch(
                    logits,
                    int(experiment["future_probability_buckets"]),
                    seed + 8_000_009,
                )
            )
            byte_values.append(piece)
            split_values.append(0 if prompt["split"] == "train" else 1)
            token_ids.append(token_id)
            state_before_rows = [
                indexed[(stage, f"state_predelta-{layer}")]
                for layer in _RECURRENT_LAYERS
            ]
            state_after_rows = [
                indexed[(stage, f"new_state-{layer}")]
                for layer in _RECURRENT_LAYERS
            ]
            metadata.append(
                {
                    "sample_id": f"{prompt['id']}:stage-{stage}",
                    "prompt_id": prompt["id"],
                    "prompt_index": prompt_index,
                    "split": prompt["split"],
                    "stage": stage,
                    "token_id": token_id,
                    "token_piece_hex": piece.hex(),
                    "prompt_sha256": hashlib.sha256(prompt["text"].encode()).hexdigest(),
                    "state_before_bundle_sha256": bundle_hash(raw_dir, state_before_rows),
                    "state_after_bundle_sha256": bundle_hash(raw_dir, state_after_rows),
                    "future_logits_sha256": sha256(logits_path),
                }
            )
        if not args.keep_raw:
            shutil.rmtree(raw_dir)

    max_bytes = max(1, max(len(value) for value in byte_values))
    emitted = np.zeros((len(byte_values), max_bytes), dtype=np.uint8)
    emitted_mask = np.zeros((len(byte_values), max_bytes), dtype=np.bool_)
    for index, value in enumerate(byte_values):
        emitted[index, : len(value)] = np.frombuffer(value, dtype=np.uint8)
        emitted_mask[index, : len(value)] = True

    arrays = {
        "state_before": np.stack(before_samples).astype(np.float32),
        "state_after": np.stack(after_samples).astype(np.float32),
        "event_features": np.stack(event_samples).astype(np.float32),
        "emitted_bytes": emitted,
        "emitted_byte_mask": emitted_mask,
        "future_probabilities": np.stack(future_samples).astype(np.float32),
        "split": np.asarray(split_values, dtype=np.uint8),
        "token_ids": np.asarray(token_ids, dtype=np.int32),
    }
    array_manifest = {
        name: write_array(temporary_output / f"{name}.npy", value)
        for name, value in arrays.items()
    }
    metadata_path = temporary_output / "samples.jsonl"
    metadata_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in metadata),
        encoding="utf-8",
    )
    split_counts = Counter(item["split"] for item in metadata)
    manifest = {
        "schema_version": 1,
        "name": experiment["name"],
        "records": len(metadata),
        "split_records": dict(sorted(split_counts.items())),
        "prompts": len(experiment["prompts"]),
        "continuation_tokens_per_prompt": continuation,
        "recurrent_layers": list(_RECURRENT_LAYERS),
        "state_feature_dim": int(experiment["state_buckets"])
        + int(experiment["conv_buckets"]),
        "event_feature_dim": int(experiment["event_features"]),
        "future_probability_buckets": int(experiment["future_probability_buckets"]),
        "max_emitted_bytes": max_bytes,
        "exact_cache_continuity": {
            "comparisons": chain_comparisons,
            "state_matches": state_chain_matches,
            "conv_matches": conv_chain_matches,
        },
        "model_sha256": sha256(model),
        "probe_source_sha256": build_report["source_sha256"],
        "probe_binary_sha256": build_report["binary_sha256"],
        "experiment_config": str(config_path),
        "experiment_config_sha256": sha256(config_path),
        "arrays": array_manifest,
        "samples": {
            "path": metadata_path.name,
            "sha256": sha256(metadata_path),
            "bytes": metadata_path.stat().st_size,
        },
        "interpretation": (
            "CountSketch projections are a bounded learnability control, not full-state "
            "reconstruction and not an acceleration claim."
        ),
    }
    manifest_path = temporary_output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (temporary_output / "README.md").write_text(
        "# Qwen3.5 projected real-state pairs v1\n\n"
        "Compact prompt-grouped learnability controls compiled from exact recurrent "
        "states of the pinned Qwen3.5 donor.\n\n"
        f"- records: {len(metadata)} ({split_counts['train']} train, "
        f"{split_counts['validation']} validation);\n"
        f"- exact cache links: {state_chain_matches}/{chain_comparisons} recurrent and "
        f"{conv_chain_matches}/{chain_comparisons} convolution;\n"
        f"- per-layer projected state: {manifest['state_feature_dim']} float32 values;\n"
        f"- future distribution: {manifest['future_probability_buckets']} probability buckets.\n\n"
        "The projections are lossy and cannot be injected as full Qwen states. See "
        "`manifest.json` and `docs/aira-qwen35-real-state-probe.md`.\n",
        encoding="utf-8",
    )
    if output.exists():
        shutil.rmtree(output)
    temporary_output.replace(output)
    return manifest


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
        "--binary", default=".cache/qwen35-state-probe/qwen35-state-probe"
    )
    parser.add_argument(
        "--build-report", default="results/qwen35_state_probe_build.json"
    )
    parser.add_argument("--work-dir", default="data/qwen35-state-pairs-work")
    parser.add_argument(
        "--output-dir", default="artifacts/qwen35-real-state-pairs-v1"
    )
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = collect(args)
    print(
        json.dumps(
            {
                "output": args.output_dir,
                "records": report["records"],
                "split_records": report["split_records"],
                "state_feature_dim": report["state_feature_dim"],
                "exact_cache_continuity": report["exact_cache_continuity"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
