#!/usr/bin/env python3
"""Export learned full recurrent matrices for native partial-state replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch

from minillm.aira.full_state import ConvRowUpdater, GatedDeltaParameterUpdater


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_tensor(
    raw_root: Path,
    sample: dict[str, Any],
    name: str,
    layer: int,
    shape: tuple[int, ...],
) -> np.ndarray:
    path = (
        raw_root / sample["prompt_id"] / f"stage-{sample['stage']}.{name}-{layer}.bin"
    )
    value = np.fromfile(path, dtype="<f4")
    if value.size != int(np.prod(shape)):
        raise ValueError(f"unexpected tensor shape: {path}")
    return value.reshape(shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default="artifacts/qwen35-gated-delta-updater-v1/model.pt"
    )
    parser.add_argument(
        "--conv-checkpoint", default="artifacts/qwen35-conv-row-updater-v1/model.pt"
    )
    parser.add_argument(
        "--source-pairs", default="artifacts/qwen35-real-state-pairs-v1"
    )
    parser.add_argument("--raw-dir", default=".cache/qwen35-full-state-work")
    parser.add_argument("--prompt-id", default="validation-en-arithmetic")
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--output",
        default="artifacts/qwen35-gated-delta-updater-v1/validation-patch.bin",
    )
    args = parser.parse_args()

    if not 0 < args.alpha <= 1:
        raise ValueError("patch alpha must lie in (0, 1]")
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    conv_checkpoint_path = Path(args.conv_checkpoint)
    conv_checkpoint = torch.load(
        conv_checkpoint_path, map_location="cpu", weights_only=True
    )
    config = checkpoint["config"]
    conv_config = conv_checkpoint["config"]
    layers = list(config["candidate_recurrent_layers"])
    if layers != list(conv_config["candidate_recurrent_layers"]):
        raise ValueError("state and convolution checkpoints target different layers")
    heads = int(config["heads"])
    width = int(config["state_width"])
    source_root = Path(args.source_pairs)
    manifest = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    samples = [
        json.loads(line)
        for line in (source_root / manifest["samples"]["path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    sample_index, sample = next(
        (index, item)
        for index, item in enumerate(samples)
        if item["prompt_id"] == args.prompt_id and item["stage"] == args.stage
    )
    event_features = np.load(
        source_root / manifest["arrays"]["event_features"]["path"],
        allow_pickle=False,
    )[sample_index]
    raw_root = Path(args.raw_dir)
    events = []
    before_values = []
    after_values = []
    conv_before_values = []
    conv_after_values = []
    for layer in layers:
        anchor = read_tensor(raw_root, sample, "l_out", layer - 1, (1024,))
        events.append(np.concatenate((event_features, anchor)))
        before_values.append(
            read_tensor(
                raw_root,
                sample,
                "state_predelta",
                layer,
                (heads, width, width),
            )
        )
        after_values.append(
            read_tensor(
                raw_root,
                sample,
                "new_state",
                layer,
                (heads, width, width),
            )
        )
        conv_before_values.append(
            read_tensor(raw_root, sample, "conv_states", layer, (6144, 3)).T
        )
        conv_after_values.append(
            read_tensor(raw_root, sample, "last_conv_states", layer, (6144, 3)).T
        )
    model = GatedDeltaParameterUpdater(
        event_dim=len(events[0]),
        layers=len(layers),
        heads=heads,
        state_width=width,
        hidden_dim=config["hidden_dim"],
        identity_dim=config["identity_dim"],
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    conv_model = ConvRowUpdater(
        event_dim=len(events[0]),
        layers=len(layers),
        row_width=conv_config["row_width"],
        hidden_dim=conv_config["hidden_dim"],
        bottleneck_dim=conv_config["bottleneck_dim"],
        identity_dim=conv_config["identity_dim"],
    )
    conv_model.load_state_dict(conv_checkpoint["model"])
    conv_model.eval()
    before = torch.from_numpy(np.stack(before_values).copy())
    conv_before = torch.from_numpy(np.stack(conv_before_values).copy())
    event_tensor = torch.from_numpy(np.asarray(events, dtype=np.float32))
    layer_ids = torch.arange(len(layers), dtype=torch.long)
    with torch.no_grad():
        parameters = model(event_tensor, layer_ids)
        predicted = model.apply(before, parameters)
        predicted = before + args.alpha * (predicted - before)
        conv_output = conv_model(conv_before[:, 2], event_tensor, layer_ids)
        predicted_conv = torch.stack(
            (conv_before[:, 1], conv_before[:, 2], conv_output.row), dim=1
        )
    after = torch.from_numpy(np.stack(after_values).copy())
    conv_after = torch.from_numpy(np.stack(conv_after_values).copy())
    copy_mse = float((before - after).square().mean())
    patch_mse = float((predicted - after).square().mean())
    conv_copy_mse = float((conv_before[:, 2] - conv_after[:, 2]).square().mean())
    conv_patch_mse = float((conv_output.row - conv_after[:, 2]).square().mean())

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        handle.write(b"AIRASTP2")
        handle.write(struct.pack("<IIII", len(layers), heads, width, args.stage))
        for index, layer in enumerate(layers):
            handle.write(struct.pack("<i", layer))
            handle.write(predicted[index].numpy().astype("<f4").tobytes())
            handle.write(predicted_conv[index].T.numpy().astype("<f4").tobytes())
    metadata = {
        "schema_version": 2,
        "format": "AIRASTP2",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "conv_checkpoint": str(conv_checkpoint_path),
        "conv_checkpoint_sha256": sha256(conv_checkpoint_path),
        "prompt_id": args.prompt_id,
        "prompt": next(
            prompt["text"]
            for prompt in json.loads(
                Path("configs/experiments/qwen35_real_state_pairs_v1.json").read_text(
                    encoding="utf-8"
                )
            )["prompts"]
            if prompt["id"] == args.prompt_id
        ),
        "stage": args.stage,
        "alpha": args.alpha,
        "consumed_token_id": sample["token_id"],
        "layers": layers,
        "heads": heads,
        "width": width,
        "copy_state_mse": copy_mse,
        "patch_state_mse": patch_mse,
        "state_mse_ratio": patch_mse / copy_mse,
        "conv_copy_new_row_mse": conv_copy_mse,
        "conv_patch_new_row_mse": conv_patch_mse,
        "conv_new_row_mse_ratio": conv_patch_mse / conv_copy_mse,
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
