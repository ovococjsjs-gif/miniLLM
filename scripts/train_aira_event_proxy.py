#!/usr/bin/env python3
"""Resume-safe <=300-step trainer for prepared AIra event shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from minillm.aira import (
    MultiByteEventLM,
    multi_byte_event_loss,
    read_event_shards,
)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def concatenate(root: Path, split: str) -> dict[str, np.ndarray]:
    batches = list(read_event_shards(root, split=split))
    if not batches:
        raise ValueError(f"event dataset has no {split} shards")
    names = (
        "contexts",
        "byte_targets",
        "target_lengths",
        "route_targets",
        "byte_supervised",
        "full_event_lengths",
    )
    return {
        name: np.concatenate([getattr(batch, name) for batch in batches])
        for name in names
    }


def tensor_rows(
    data: dict[str, np.ndarray], indices: np.ndarray
) -> dict[str, torch.Tensor]:
    return {
        "contexts": torch.from_numpy(data["contexts"][indices]).long(),
        "byte_targets": torch.from_numpy(data["byte_targets"][indices]).long(),
        "target_lengths": torch.from_numpy(data["target_lengths"][indices]).long(),
        "route_targets": torch.from_numpy(data["route_targets"][indices]).long(),
        "byte_supervised": torch.from_numpy(data["byte_supervised"][indices]).bool(),
    }


def evaluate(
    model: MultiByteEventLM,
    data: dict[str, np.ndarray],
    *,
    batch_size: int = 1024,
) -> dict[str, float]:
    totals = {"total": 0.0, "byte": 0.0, "continuation": 0.0, "route": 0.0}
    count = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(data["contexts"]), batch_size):
            indices = np.arange(start, min(start + batch_size, len(data["contexts"])))
            batch = tensor_rows(data, indices)
            losses = multi_byte_event_loss(
                model(batch["contexts"]),
                batch["byte_targets"],
                batch["target_lengths"],
                batch["route_targets"],
                byte_supervised=batch["byte_supervised"],
            )
            size = len(indices)
            for name in totals:
                totals[name] += float(getattr(losses, name)) * size
            count += size
    return {name: value / count for name, value in totals.items()}


def save_checkpoint(
    path: Path,
    *,
    model: MultiByteEventLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict[str, Any],
    config_sha256: str,
    dataset_manifest_sha256: str,
    rng: np.random.Generator,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 1,
            "step": step,
            "config": config,
            "config_sha256": config_sha256,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": rng.bit_generator.state,
        },
        temporary,
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-data", required=True)
    parser.add_argument(
        "--config", default="configs/experiments/aira_event_pretrain_v1.json"
    )
    parser.add_argument("--steps", type=int, default=300, choices=range(1, 301))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--output", default="results/aira_event_training_proxy.json")
    args = parser.parse_args()
    root = Path(args.event_data)
    config_path = Path(args.config)
    manifest_path = root / "manifest.json"
    config = json.loads(config_path.read_text())
    model_config = config["model"]
    training_config = config["training"]
    config_hash = sha256(config_path)
    manifest_hash = sha256(manifest_path)
    train = concatenate(root, "train")
    validation = concatenate(root, "validation")
    if train["contexts"].shape[1] != model_config["dynamic_bpe_context_tokens"]:
        raise ValueError("event data context width differs from model config")
    if train["byte_targets"].shape[1] != model_config["maximum_output_bytes"]:
        raise ValueError("event data literal width differs from model config")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = MultiByteEventLM(
        vocab_size=model_config["vocab_size"],
        context_size=model_config["dynamic_bpe_context_tokens"],
        d_model=model_config["d_model"],
        maximum_bytes=model_config["maximum_output_bytes"],
        route_actions=model_config["route_actions"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    first_step = 0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if payload["config_sha256"] != config_hash:
            raise ValueError("resume config identity differs")
        if payload["dataset_manifest_sha256"] != manifest_hash:
            raise ValueError("resume event dataset identity differs")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["torch_rng_state"])
        rng.bit_generator.state = payload["numpy_rng_state"]
        first_step = int(payload["step"])
        if first_step >= args.steps:
            raise ValueError("resume checkpoint already reached requested steps")

    batch_size = int(training_config["batch_size"])
    samples = []
    started = time.perf_counter()
    model.train()
    for step in range(first_step, args.steps):
        indices = rng.integers(0, len(train["contexts"]), size=batch_size)
        batch = tensor_rows(train, indices)
        losses = multi_byte_event_loss(
            model(batch["contexts"]),
            batch["byte_targets"],
            batch["target_lengths"],
            batch["route_targets"],
            byte_supervised=batch["byte_supervised"],
            continuation_weight=training_config["multi_byte_continuation_weight"],
            route_weight=training_config["route_weight"],
        )
        if not torch.isfinite(losses.total):
            raise FloatingPointError(f"non-finite event loss at step {step + 1}")
        optimizer.zero_grad(set_to_none=True)
        losses.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in {first_step, (first_step + args.steps) // 2, args.steps - 1}:
            samples.append(
                {
                    "step": step + 1,
                    "total": float(losses.total.detach()),
                    "byte": float(losses.byte.detach()),
                    "continuation": float(losses.continuation.detach()),
                    "route": float(losses.route.detach()),
                }
            )
    training_seconds = time.perf_counter() - started
    report = {
        "schema_version": 1,
        "experiment": "aira-v2-event-trainer-smoke-v1",
        "warning": "Local <=300-step plumbing run; source-copy pointer quality and autonomous generation are separate gates.",
        "config": str(config_path),
        "config_sha256": config_hash,
        "event_data": str(root),
        "event_manifest_sha256": manifest_hash,
        "seed": args.seed,
        "steps": args.steps,
        "train_events": len(train["contexts"]),
        "validation_events": len(validation["contexts"]),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_seconds": training_seconds,
        "events_per_second": (args.steps - first_step) * batch_size / training_seconds,
        "loss_samples": samples,
        "validation": evaluate(model, validation),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    if args.checkpoint:
        save_checkpoint(
            Path(args.checkpoint),
            model=model,
            optimizer=optimizer,
            step=args.steps,
            config=config,
            config_sha256=config_hash,
            dataset_manifest_sha256=manifest_hash,
            rng=rng,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
