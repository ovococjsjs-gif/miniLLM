#!/usr/bin/env python3
"""Train a bounded patcher control on projected real Qwen recurrent states."""

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
from torch import nn
from torch.nn import functional as F

from minillm.aira.state_patcher import RecurrentStatePatcher, state_patch_loss


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class FutureBucketReadout(nn.Module):
    def __init__(self, layers: int, state_dim: int, buckets: int) -> None:
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(layers))
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.SiLU(),
            nn.Linear(128, buckets),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.layer_logits, dim=0).view(1, -1, 1)
        return self.network((state * weights).sum(dim=1))


def load_array(root: Path, manifest: dict[str, Any], name: str) -> np.ndarray:
    metadata = manifest["arrays"][name]
    path = root / metadata["path"]
    if sha256(path) != metadata["sha256"]:
        raise ValueError(f"dataset array hash mismatch: {name}")
    value = np.load(path, allow_pickle=False)
    if list(value.shape) != metadata["shape"] or str(value.dtype) != metadata["dtype"]:
        raise ValueError(f"dataset array schema mismatch: {name}")
    return value


def kl_to_probabilities(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.kl_div(F.log_softmax(logits, dim=-1), target, reduction="batchmean")


def evaluate(
    patcher: RecurrentStatePatcher,
    readout: FutureBucketReadout,
    before: torch.Tensor,
    after: torch.Tensor,
    event: torch.Tensor,
    emitted: torch.Tensor,
    emitted_mask: torch.Tensor,
    future: torch.Tensor,
) -> dict[str, float]:
    patch_mask = torch.ones(
        (before.shape[0], before.shape[1]), dtype=torch.bool, device=before.device
    )
    with torch.no_grad():
        output = patcher(
            before,
            event,
            emitted,
            patch_mask=patch_mask,
            emitted_byte_mask=emitted_mask,
        )
        copy_mse = F.mse_loss(before, after)
        patch_mse = F.mse_loss(output.state, after)
        copy_cosine = (
            1 - F.cosine_similarity(before.flatten(0, 1), after.flatten(0, 1), dim=-1)
        ).mean()
        patch_cosine = (
            1
            - F.cosine_similarity(
                output.state.flatten(0, 1), after.flatten(0, 1), dim=-1
            )
        ).mean()
        oracle_future_kl = kl_to_probabilities(readout(after), future)
        copy_future_kl = kl_to_probabilities(readout(before), future)
        patch_future_kl = kl_to_probabilities(readout(output.state), future)
    return {
        "records": float(before.shape[0]),
        "copy_state_mse": float(copy_mse),
        "patch_state_mse": float(patch_mse),
        "state_mse_ratio": float(patch_mse / copy_mse),
        "copy_state_cosine_error": float(copy_cosine),
        "patch_state_cosine_error": float(patch_cosine),
        "oracle_future_kl": float(oracle_future_kl),
        "copy_future_kl": float(copy_future_kl),
        "patch_future_kl": float(patch_future_kl),
        "future_kl_improvement": float(copy_future_kl - patch_future_kl),
        "mean_confidence": float(output.confidence.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="artifacts/qwen35-real-state-pairs-v1"
    )
    parser.add_argument("--readout-steps", type=int, default=100)
    parser.add_argument("--patcher-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--cosine-weight", type=float, default=0.2)
    parser.add_argument("--future-kl-weight", type=float, default=0.0)
    parser.add_argument("--confidence-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=811)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--checkpoint", default="artifacts/qwen35-real-state-patcher-v1/model.pt"
    )
    parser.add_argument(
        "--output", default="results/qwen35_real_state_patcher_proxy.json"
    )
    args = parser.parse_args()
    if args.readout_steps < 1 or args.patcher_steps < 1:
        raise ValueError("training step counts must be positive")
    if args.readout_steps + args.patcher_steps > 300:
        raise ValueError("combined local training may not exceed 300 steps")

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    root = Path(args.dataset)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    before = torch.from_numpy(load_array(root, manifest, "state_before").copy())
    after = torch.from_numpy(load_array(root, manifest, "state_after").copy())
    event = torch.from_numpy(load_array(root, manifest, "event_features").copy())
    emitted = torch.from_numpy(load_array(root, manifest, "emitted_bytes").copy()).long()
    emitted_mask = torch.from_numpy(
        load_array(root, manifest, "emitted_byte_mask").copy()
    ).bool()
    future = torch.from_numpy(
        load_array(root, manifest, "future_probabilities").copy()
    )
    split = torch.from_numpy(load_array(root, manifest, "split").copy())
    train_indices = torch.nonzero(split == 0, as_tuple=False).squeeze(-1)
    validation_indices = torch.nonzero(split == 1, as_tuple=False).squeeze(-1)
    if train_indices.numel() < 1 or validation_indices.numel() < 1:
        raise ValueError("real-state dataset needs train and validation prompt groups")

    train_states = torch.cat((before[train_indices], after[train_indices]), dim=0)
    state_mean = train_states.mean(dim=0, keepdim=True)
    state_std = train_states.std(dim=0, keepdim=True).clamp_min(1e-5)
    before = (before - state_mean) / state_std
    after = (after - state_mean) / state_std
    mean_train_delta = (after[train_indices] - before[train_indices]).mean(
        dim=0, keepdim=True
    )
    validation_copy_mse = F.mse_loss(
        before[validation_indices], after[validation_indices]
    )
    validation_mean_delta_mse = F.mse_loss(
        before[validation_indices] + mean_train_delta,
        after[validation_indices],
    )

    layers = before.shape[1]
    state_dim = before.shape[2]
    event_dim = event.shape[1]
    future_buckets = future.shape[1]
    readout = FutureBucketReadout(layers, state_dim, future_buckets)
    readout_optimizer = torch.optim.AdamW(
        readout.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    readout_samples = []
    started = time.perf_counter()
    readout.train()
    for step in range(args.readout_steps):
        batch = torch.tensor(
            [rng.choice(train_indices.tolist()) for _ in range(args.batch_size)]
        )
        loss = kl_to_probabilities(readout(after[batch]), future[batch])
        readout_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(readout.parameters(), 1.0)
        readout_optimizer.step()
        if step in {0, args.readout_steps // 2, args.readout_steps - 1}:
            readout_samples.append(
                {
                    "step": step + 1,
                    "kl": float(loss.detach()),
                    "gradient_norm": float(gradient_norm),
                }
            )
    for parameter in readout.parameters():
        parameter.requires_grad_(False)
    readout.eval()

    patcher = RecurrentStatePatcher(
        layers=layers,
        state_dim=state_dim,
        event_dim=event_dim,
        hidden_dim=128,
        byte_dim=16,
    )
    patcher_optimizer = torch.optim.AdamW(
        patcher.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    patcher_samples = []
    patcher.train()
    for step in range(args.patcher_steps):
        batch = torch.tensor(
            [rng.choice(train_indices.tolist()) for _ in range(args.batch_size)]
        )
        patch_mask = torch.ones((args.batch_size, layers), dtype=torch.bool)
        output = patcher(
            before[batch],
            event[batch],
            emitted[batch],
            patch_mask=patch_mask,
            emitted_byte_mask=emitted_mask[batch],
        )
        student_future = readout(output.state)
        teacher_future_logits = torch.log(future[batch].clamp_min(1e-12))
        loss = state_patch_loss(
            output,
            after[batch],
            patch_mask,
            student_future_logits=student_future,
            teacher_future_logits=teacher_future_logits,
            cosine_weight=args.cosine_weight,
            future_kl_weight=args.future_kl_weight,
            confidence_weight=args.confidence_weight,
        )
        patcher_optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(patcher.parameters(), 1.0)
        patcher_optimizer.step()
        if step in {0, args.patcher_steps // 4, args.patcher_steps // 2, 3 * args.patcher_steps // 4, args.patcher_steps - 1}:
            patcher_samples.append(
                {
                    "step": step + 1,
                    "total": float(loss.total.detach()),
                    "state_mse": float(loss.state_mse.detach()),
                    "state_cosine": float(loss.state_cosine.detach()),
                    "future_kl": float(loss.future_kl.detach()),
                    "confidence_bce": float(loss.confidence_bce.detach()),
                    "gradient_norm": float(gradient_norm),
                }
            )
    training_seconds = time.perf_counter() - started
    patcher.eval()
    train_metrics = evaluate(
        patcher,
        readout,
        before[train_indices],
        after[train_indices],
        event[train_indices],
        emitted[train_indices],
        emitted_mask[train_indices],
        future[train_indices],
    )
    validation_metrics = evaluate(
        patcher,
        readout,
        before[validation_indices],
        after[validation_indices],
        event[validation_indices],
        emitted[validation_indices],
        emitted_mask[validation_indices],
        future[validation_indices],
    )

    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "patcher": patcher.state_dict(),
            "readout": readout.state_dict(),
            "state_mean": state_mean,
            "state_std": state_std,
            "dataset_manifest_sha256": sha256(manifest_path),
            "seed": args.seed,
        },
        checkpoint,
    )
    report = {
        "schema_version": 1,
        "experiment": "qwen35-projected-real-state-patcher-v1",
        "warning": (
            "This is a CountSketch/probability-bucket learnability control. It is not "
            "full-state reconstruction and does not authorize layer skipping."
        ),
        "dataset": str(root),
        "dataset_manifest_sha256": sha256(manifest_path),
        "records": manifest["records"],
        "train_records": int(train_indices.numel()),
        "validation_records": int(validation_indices.numel()),
        "layers": layers,
        "state_dim": state_dim,
        "event_dim": event_dim,
        "future_probability_buckets": future_buckets,
        "readout_steps": args.readout_steps,
        "patcher_steps": args.patcher_steps,
        "total_steps": args.readout_steps + args.patcher_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "loss_weights": {
            "cosine": args.cosine_weight,
            "future_kl": args.future_kl_weight,
            "confidence": args.confidence_weight,
        },
        "seed": args.seed,
        "training_seconds": training_seconds,
        "patcher_parameters": sum(parameter.numel() for parameter in patcher.parameters()),
        "readout_parameters": sum(parameter.numel() for parameter in readout.parameters()),
        "readout_loss_samples": readout_samples,
        "patcher_loss_samples": patcher_samples,
        "validation_baselines": {
            "copy_state_mse": float(validation_copy_mse),
            "mean_train_delta_state_mse": float(validation_mean_delta_mse),
            "mean_delta_over_copy_ratio": float(
                validation_mean_delta_mse / validation_copy_mse
            ),
        },
        "train": train_metrics,
        "validation": validation_metrics,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "acceptance": {
            "beats_copy_state_mse": validation_metrics["state_mse_ratio"] < 1.0,
            "beats_mean_delta_state_mse": (
                validation_metrics["patch_state_mse"]
                < float(validation_mean_delta_mse)
            ),
            "beats_copy_future_kl": validation_metrics["future_kl_improvement"] > 0.0,
            "full_state_gate_passed": False,
            "generated_quality_gate_passed": False,
            "acceleration_claim_allowed": False,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
