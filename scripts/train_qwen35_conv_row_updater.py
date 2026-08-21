#!/usr/bin/env python3
"""Train the missing newest-row predictor for Qwen convolution caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from train_qwen35_gated_delta_updater import conditioning, read_tensor, verified_array

from minillm.aira.full_state import ConvRowUpdater


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_batch(
    raw_root: Path,
    samples: list[dict],
    token_features: np.ndarray,
    layers: list[int],
    pairs: list[tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    previous = []
    target = []
    events = []
    layer_ids = []
    for sample_index, layer_index in pairs:
        sample = samples[sample_index]
        layer = layers[layer_index]
        before = read_tensor(raw_root, sample, "conv_states", layer, (6144, 3)).T
        after = read_tensor(raw_root, sample, "last_conv_states", layer, (6144, 3)).T
        if not np.array_equal(after[0], before[1]) or not np.array_equal(
            after[1], before[2]
        ):
            raise ValueError("convolution cache did not follow its exact two-row shift")
        previous.append(before[2])
        target.append(after[2])
        events.append(
            conditioning(raw_root, sample, layer, token_features[sample_index])
        )
        layer_ids.append(layer_index)
    return (
        torch.from_numpy(np.stack(previous).copy()),
        torch.from_numpy(np.stack(target).copy()),
        torch.from_numpy(np.stack(events).copy()),
        torch.tensor(layer_ids, dtype=torch.long),
    )


def evaluate(
    model: ConvRowUpdater,
    raw_root: Path,
    samples: list[dict],
    token_features: np.ndarray,
    layers: list[int],
    indices: list[int],
) -> dict[str, float]:
    copy_squared = 0.0
    patch_squared = 0.0
    elements = 0
    cosine_copy = []
    cosine_patch = []
    confidences = []
    with torch.no_grad():
        for sample_index in indices:
            pairs = [(sample_index, layer_index) for layer_index in range(len(layers))]
            previous, target, event, layer_ids = load_batch(
                raw_root, samples, token_features, layers, pairs
            )
            output = model(previous, event, layer_ids)
            copy_squared += float((target - previous).square().sum())
            patch_squared += float((target - output.row).square().sum())
            elements += target.numel()
            cosine_copy.extend(
                (1 - F.cosine_similarity(previous, target, dim=-1)).tolist()
            )
            cosine_patch.extend(
                (1 - F.cosine_similarity(output.row, target, dim=-1)).tolist()
            )
            confidences.extend(output.confidence.tolist())
    copy_mse = copy_squared / elements
    patch_mse = patch_squared / elements
    return {
        "transitions": float(len(indices)),
        "layer_rows": float(len(indices) * len(layers)),
        "copy_new_row_mse": copy_mse,
        "patch_new_row_mse": patch_mse,
        "new_row_mse_ratio": patch_mse / copy_mse,
        "copy_cosine_error": float(np.mean(cosine_copy)),
        "patch_cosine_error": float(np.mean(cosine_patch)),
        "mean_confidence": float(np.mean(confidences)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/experiments/qwen35_conv_row_updater_v1.json"
    )
    parser.add_argument(
        "--checkpoint", default="artifacts/qwen35-conv-row-updater-v1/model.pt"
    )
    parser.add_argument("--output", default="results/qwen35_conv_row_updater_v1.json")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["steps"] < 1 or config["steps"] > 300:
        raise ValueError("convolution updater training must use 1..300 steps")
    source_root = Path(config["source_pairs"])
    manifest_path = source_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    token_features = verified_array(source_root, manifest, "event_features")
    split = verified_array(source_root, manifest, "split")
    samples = [
        json.loads(line)
        for line in (source_root / manifest["samples"]["path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    raw_root = Path(config["raw_state_work_dir"])
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw transition tensors are absent: {raw_root}")
    layers = list(config["candidate_recurrent_layers"])
    train_indices = np.flatnonzero(split == 0).tolist()
    validation_indices = np.flatnonzero(split == 1).tolist()

    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    rng = random.Random(config["seed"])
    model = ConvRowUpdater(
        event_dim=token_features.shape[1] + config["anchor_hidden_dim"],
        layers=len(layers),
        row_width=config["row_width"],
        hidden_dim=config["hidden_dim"],
        bottleneck_dim=config["bottleneck_dim"],
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
        previous, target, event, layer_ids = load_batch(
            raw_root, samples, token_features, layers, pairs
        )
        output = model(previous, event, layer_ids)
        copy_mse = (target - previous).square().mean(dim=1).clamp_min(1e-10)
        patch_mse = (target - output.row).square().mean(dim=1)
        ratios = patch_mse / copy_mse
        confidence_target = torch.exp(-ratios.detach()).clamp(0, 1)
        confidence_loss = F.binary_cross_entropy(output.confidence, confidence_target)
        loss = ratios.mean() + config["confidence_weight"] * confidence_loss
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
                    "mse_ratio": float(ratios.mean().detach()),
                    "copy_mse": float(copy_mse.mean().detach()),
                    "patch_mse": float(patch_mse.mean().detach()),
                    "confidence_bce": float(confidence_loss.detach()),
                    "gradient_norm": float(gradient_norm),
                }
            )
    training_seconds = time.perf_counter() - started
    model.eval()
    train_metrics = evaluate(
        model, raw_root, samples, token_features, layers, train_indices
    )
    validation_metrics = evaluate(
        model, raw_root, samples, token_features, layers, validation_indices
    )

    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "model": model.state_dict(),
            "config": config,
            "config_sha256": sha256(config_path),
            "source_manifest_sha256": sha256(manifest_path),
        },
        checkpoint,
    )
    threshold = config["acceptance"]["validation_new_row_mse_ratio_below"]
    report = {
        "schema_version": 1,
        "experiment": config["name"],
        "role": "learned newest-row cache updater; not an acceleration claim",
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "source_manifest_sha256": sha256(manifest_path),
        "candidate_recurrent_layers": layers,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "steps": config["steps"],
        "training_seconds": training_seconds,
        "loss_samples": loss_samples,
        "train": train_metrics,
        "validation": validation_metrics,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "acceptance": {
            "older_rows_shift_exact": True,
            "beats_copy_new_row_mse": validation_metrics["new_row_mse_ratio"]
            < threshold,
            "injected_future_kl_gate_passed": False,
            "deployment_allowed": False,
        },
        "limitations": (
            "The result must be combined with learned full recurrent states and replayed "
            "through Qwen before any layer-skip or speed claim."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
