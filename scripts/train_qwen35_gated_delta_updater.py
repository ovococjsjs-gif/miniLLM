#!/usr/bin/env python3
"""Train an injectible updater using Qwen's stable Gated DeltaNet algebra."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from minillm.aira.full_state import (
    GatedDeltaParameters,
    GatedDeltaParameterUpdater,
)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verified_array(root: Path, manifest: dict[str, Any], name: str) -> np.ndarray:
    metadata = manifest["arrays"][name]
    path = root / metadata["path"]
    if sha256(path) != metadata["sha256"]:
        raise ValueError(f"source pair array hash mismatch: {name}")
    value = np.load(path, allow_pickle=False)
    if list(value.shape) != metadata["shape"] or str(value.dtype) != metadata["dtype"]:
        raise ValueError(f"source pair array schema mismatch: {name}")
    return value


def tensor_path(raw_root: Path, sample: dict[str, Any], name: str, layer: int) -> Path:
    return (
        raw_root / sample["prompt_id"] / f"stage-{sample['stage']}.{name}-{layer}.bin"
    )


def read_tensor(
    raw_root: Path,
    sample: dict[str, Any],
    name: str,
    layer: int,
    shape: tuple[int, ...],
) -> np.ndarray:
    path = tensor_path(raw_root, sample, name, layer)
    value = np.fromfile(path, dtype="<f4")
    if value.size != int(np.prod(shape)):
        raise ValueError(f"unexpected tensor shape for {path}")
    return value.reshape(shape)


def conditioning(
    raw_root: Path,
    sample: dict[str, Any],
    layer: int,
    token_features: np.ndarray,
) -> np.ndarray:
    if layer < 1:
        raise ValueError("candidate recurrent layer needs a preceding anchor")
    anchor = read_tensor(raw_root, sample, "l_out", layer - 1, (1024,))
    return np.concatenate((token_features, anchor)).astype(np.float32)


def target_parameters(
    raw_root: Path,
    sample: dict[str, Any],
    layer: int,
    *,
    heads: int,
    width: int,
) -> GatedDeltaParameters:
    key = torch.from_numpy(
        read_tensor(raw_root, sample, "k_conv_predelta", layer, (heads, width)).copy()
    )
    value = torch.from_numpy(
        read_tensor(raw_root, sample, "v_conv_predelta", layer, (heads, width)).copy()
    )
    gate = torch.from_numpy(
        read_tensor(raw_root, sample, "gate", layer, (heads,)).copy()
    )
    beta = torch.from_numpy(
        read_tensor(raw_root, sample, "beta_sigmoid", layer, (heads,)).copy()
    )
    return GatedDeltaParameters(key, value, gate, beta)


def stack_parameters(items: list[GatedDeltaParameters]) -> GatedDeltaParameters:
    return GatedDeltaParameters(
        *(
            torch.stack([getattr(item, field) for item in items])
            for field in (
                "key",
                "value",
                "gate",
                "beta",
            )
        )
    )


def load_batch(
    raw_root: Path,
    samples: list[dict[str, Any]],
    token_features: np.ndarray,
    layers: list[int],
    pairs: list[tuple[int, int]],
    *,
    heads: int,
    width: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    GatedDeltaParameters,
]:
    before = []
    after = []
    events = []
    layer_ids = []
    parameters = []
    for sample_index, layer_index in pairs:
        sample = samples[sample_index]
        layer = layers[layer_index]
        before.append(
            read_tensor(
                raw_root,
                sample,
                "state_predelta",
                layer,
                (heads, width, width),
            )
        )
        after.append(
            read_tensor(
                raw_root,
                sample,
                "new_state",
                layer,
                (heads, width, width),
            )
        )
        events.append(
            conditioning(raw_root, sample, layer, token_features[sample_index])
        )
        layer_ids.append(layer_index)
        parameters.append(
            target_parameters(raw_root, sample, layer, heads=heads, width=width)
        )
    return (
        torch.from_numpy(np.stack(before).copy()),
        torch.from_numpy(np.stack(after).copy()),
        torch.from_numpy(np.stack(events).copy()),
        torch.tensor(layer_ids, dtype=torch.long),
        stack_parameters(parameters),
    )


def transition_losses(
    predicted: GatedDeltaParameters,
    target: GatedDeltaParameters,
    predicted_state: torch.Tensor,
    source_state: torch.Tensor,
    target_state: torch.Tensor,
) -> dict[str, torch.Tensor]:
    key_cosine = (1 - F.cosine_similarity(predicted.key, target.key, dim=-1)).mean()
    value_scale = target.value.square().mean(dim=(1, 2)).clamp_min(1e-8)
    value_ratio = (
        (predicted.value - target.value)
        .square()
        .mean(dim=(1, 2))
        .div(value_scale)
        .mean()
    )
    gate_scale = target.gate.square().mean(dim=1).clamp_min(1e-8)
    gate_ratio = (
        (predicted.gate - target.gate).square().mean(dim=1).div(gate_scale).mean()
    )
    beta_mse = (predicted.beta - target.beta).square().mean()
    copy_mse = (target_state - source_state).square().mean(dim=(1, 2, 3))
    patch_mse = (target_state - predicted_state).square().mean(dim=(1, 2, 3))
    state_ratio = (patch_mse / copy_mse.clamp_min(1e-10)).mean()
    return {
        "key_cosine": key_cosine,
        "value_ratio": value_ratio,
        "gate_ratio": gate_ratio,
        "beta_mse": beta_mse,
        "copy_mse": copy_mse.mean(),
        "patch_mse": patch_mse.mean(),
        "state_ratio": state_ratio,
    }


def structure_controls(
    raw_root: Path,
    samples: list[dict[str, Any]],
    token_features: np.ndarray,
    layers: list[int],
    *,
    heads: int,
    width: int,
) -> dict[str, Any]:
    maximum = 0.0
    squared = 0.0
    elements = 0
    conv_matches = 0
    conv_comparisons = 0
    low_rank_residuals: dict[int, list[float]] = {1: [], 2: [], 4: [], 8: []}
    started = time.perf_counter()
    with torch.no_grad():
        for sample_index, sample in enumerate(samples):
            for layer_index, layer in enumerate(layers):
                before, after, _, _, target = load_batch(
                    raw_root,
                    samples,
                    token_features,
                    layers,
                    [(sample_index, layer_index)],
                    heads=heads,
                    width=width,
                )
                reconstructed = GatedDeltaParameterUpdater.apply(before, target)
                difference = reconstructed - after
                maximum = max(maximum, float(difference.abs().max()))
                squared += float(difference.square().sum())
                elements += difference.numel()
                delta = (after - before).numpy()[0]
                for head in (0, 5, 10, 15):
                    singular = np.linalg.svd(delta[head], compute_uv=False)
                    energy = np.square(singular.astype(np.float64))
                    total = max(float(energy.sum()), 1e-30)
                    for rank, residuals in low_rank_residuals.items():
                        residuals.append(float(energy[rank:].sum()) / total)
                conv_before = read_tensor(
                    raw_root, sample, "conv_states", layer, (6144, 3)
                ).T
                conv_after = read_tensor(
                    raw_root, sample, "last_conv_states", layer, (6144, 3)
                ).T
                conv_comparisons += 2
                conv_matches += int(np.array_equal(conv_after[0], conv_before[1]))
                conv_matches += int(np.array_equal(conv_after[1], conv_before[2]))
    return {
        "exact_formula_max_abs": maximum,
        "exact_formula_mse": squared / elements,
        "exact_formula_elements": elements,
        "conv_shift_matches": conv_matches,
        "conv_shift_comparisons": conv_comparisons,
        "sampled_delta_residual_energy": {
            f"rank_{rank}": float(np.mean(values))
            for rank, values in low_rank_residuals.items()
        },
        "seconds": time.perf_counter() - started,
    }


def evaluate(
    model: GatedDeltaParameterUpdater,
    raw_root: Path,
    samples: list[dict[str, Any]],
    token_features: np.ndarray,
    layers: list[int],
    sample_indices: list[int],
    *,
    heads: int,
    width: int,
) -> dict[str, float]:
    totals = {
        "copy_squared": 0.0,
        "patch_squared": 0.0,
        "elements": 0.0,
        "key_cosine": 0.0,
        "value_ratio": 0.0,
        "gate_ratio": 0.0,
        "beta_mse": 0.0,
        "pairs": 0.0,
    }
    with torch.no_grad():
        for sample_index in sample_indices:
            for layer_index in range(len(layers)):
                before, after, event, layer_ids, target = load_batch(
                    raw_root,
                    samples,
                    token_features,
                    layers,
                    [(sample_index, layer_index)],
                    heads=heads,
                    width=width,
                )
                predicted = model(event, layer_ids)
                predicted_state = model.apply(before, predicted)
                metrics = transition_losses(
                    predicted, target, predicted_state, before, after
                )
                totals["copy_squared"] += float((after - before).square().sum())
                totals["patch_squared"] += float(
                    (after - predicted_state).square().sum()
                )
                totals["elements"] += after.numel()
                for name in ("key_cosine", "value_ratio", "gate_ratio", "beta_mse"):
                    totals[name] += float(metrics[name])
                totals["pairs"] += 1
    copy_mse = totals["copy_squared"] / totals["elements"]
    patch_mse = totals["patch_squared"] / totals["elements"]
    return {
        "transitions": float(len(sample_indices)),
        "layer_transitions": totals["pairs"],
        "copy_state_mse": copy_mse,
        "patch_state_mse": patch_mse,
        "state_mse_ratio": patch_mse / copy_mse,
        **{
            name: totals[name] / totals["pairs"]
            for name in ("key_cosine", "value_ratio", "gate_ratio", "beta_mse")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/experiments/qwen35_gated_delta_updater_v1.json"
    )
    parser.add_argument(
        "--checkpoint", default="artifacts/qwen35-gated-delta-updater-v1/model.pt"
    )
    parser.add_argument(
        "--output", default="results/qwen35_gated_delta_updater_v1.json"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["steps"] < 1 or config["steps"] > 300:
        raise ValueError("gated-delta training must use 1..300 steps")
    source_root = Path(config["source_pairs"])
    source_manifest_path = source_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    token_features = verified_array(source_root, source_manifest, "event_features")
    split = verified_array(source_root, source_manifest, "split")
    samples = [
        json.loads(line)
        for line in (source_root / source_manifest["samples"]["path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    raw_root = Path(config["raw_state_work_dir"])
    if not raw_root.is_dir():
        raise FileNotFoundError(
            f"raw transition tensors are absent: {raw_root}; recollect with transition probe"
        )
    build_path = Path(config["transition_probe_build"])
    build = json.loads(build_path.read_text(encoding="utf-8"))
    if build["source"] != "native/qwen35_transition_probe.cpp":
        raise ValueError("raw transition probe provenance is not the expected source")
    layers = list(config["candidate_recurrent_layers"])
    heads = int(config["heads"])
    width = int(config["state_width"])
    train_indices = np.flatnonzero(split == 0).tolist()
    validation_indices = np.flatnonzero(split == 1).tolist()
    if len(train_indices) != 32 or len(validation_indices) != 16:
        raise ValueError("expected the pinned 32/16 prompt-group transition split")

    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    rng = random.Random(config["seed"])
    structure = structure_controls(
        raw_root,
        samples,
        token_features,
        layers,
        heads=heads,
        width=width,
    )
    model = GatedDeltaParameterUpdater(
        event_dim=token_features.shape[1] + config["anchor_hidden_dim"],
        layers=len(layers),
        heads=heads,
        state_width=width,
        hidden_dim=config["hidden_dim"],
        identity_dim=config["identity_dim"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    loss_samples = []
    started = time.perf_counter()
    model.train()
    for step in range(config["steps"]):
        pairs = [
            (rng.choice(train_indices), rng.randrange(len(layers)))
            for _ in range(config["batch_size"])
        ]
        before, after, event, layer_ids, target = load_batch(
            raw_root,
            samples,
            token_features,
            layers,
            pairs,
            heads=heads,
            width=width,
        )
        predicted = model(event, layer_ids)
        predicted_state = model.apply(before, predicted)
        metrics = transition_losses(predicted, target, predicted_state, before, after)
        loss = (
            config["key_loss_weight"] * metrics["key_cosine"]
            + config["value_loss_weight"] * metrics["value_ratio"]
            + config["gate_loss_weight"] * metrics["gate_ratio"]
            + config["beta_loss_weight"] * metrics["beta_mse"]
            + config["state_loss_weight"] * metrics["state_ratio"]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in {
            0,
            config["steps"] // 4,
            config["steps"] // 2,
            3 * config["steps"] // 4,
            config["steps"] - 1,
        }:
            loss_samples.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "state_ratio": float(metrics["state_ratio"].detach()),
                    "key_cosine": float(metrics["key_cosine"].detach()),
                    "value_ratio": float(metrics["value_ratio"].detach()),
                    "gate_ratio": float(metrics["gate_ratio"].detach()),
                    "beta_mse": float(metrics["beta_mse"].detach()),
                    "gradient_norm": float(gradient_norm),
                }
            )
    training_seconds = time.perf_counter() - started
    model.eval()
    train_metrics = evaluate(
        model,
        raw_root,
        samples,
        token_features,
        layers,
        train_indices,
        heads=heads,
        width=width,
    )
    validation_metrics = evaluate(
        model,
        raw_root,
        samples,
        token_features,
        layers,
        validation_indices,
        heads=heads,
        width=width,
    )

    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "model": model.state_dict(),
            "config": config,
            "config_sha256": sha256(config_path),
            "source_manifest_sha256": sha256(source_manifest_path),
            "transition_probe_source_sha256": build["source_sha256"],
        },
        checkpoint,
    )
    acceptance = config["acceptance"]
    report = {
        "schema_version": 1,
        "experiment": config["name"],
        "role": (
            "learned stable-parameter updater producing full injectible Qwen recurrent "
            "states; not yet an acceleration claim"
        ),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "source_manifest_sha256": sha256(source_manifest_path),
        "transition_probe_build": str(build_path),
        "transition_probe_source_sha256": build["source_sha256"],
        "raw_state_bytes": sum(path.stat().st_size for path in raw_root.rglob("*.bin")),
        "candidate_recurrent_layers": layers,
        "conditioning": config["conditioning"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "steps": config["steps"],
        "training_seconds": training_seconds,
        "structure": structure,
        "loss_samples": loss_samples,
        "train": train_metrics,
        "validation": validation_metrics,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "acceptance": {
            "full_state_formula_verified": structure["exact_formula_max_abs"]
            < acceptance["exact_formula_max_abs_below"],
            "conv_shift_structure_exact": structure["conv_shift_matches"]
            == structure["conv_shift_comparisons"],
            "beats_copy_state_mse": validation_metrics["state_mse_ratio"]
            < acceptance["validation_state_mse_ratio_below"],
            "injected_future_logit_gate_passed": False,
            "generated_quality_gate_passed": False,
            "measured_speedup_gate_passed": False,
            "deployment_allowed": False,
        },
        "limitations": (
            "The updater uses a computed attention-anchor hidden state and predicts exact "
            "Gated DeltaNet parameters. Native injection, convolution new-row prediction, "
            "future logits, free generation, and measured skip speed remain mandatory."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (checkpoint.parent / "README.md").write_text(
        "# Qwen3.5 Gated DeltaNet updater v1\n\n"
        "Predicts key/value/gate/beta for five recurrent layers immediately following "
        "attention anchors, then reconstructs exact-size 16x128x128 states using Qwen's "
        "native transition algebra. Deployment remains disabled until injection, "
        "generation, convolution, and speed gates pass.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "parameters": report["parameters"],
                "steps": report["steps"],
                "structure": structure,
                "train": train_metrics,
                "validation": validation_metrics,
                "acceptance": report["acceptance"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
