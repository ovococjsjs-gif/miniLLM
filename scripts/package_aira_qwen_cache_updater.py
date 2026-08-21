#!/usr/bin/env python3
"""Merge normalized state/conv components into one inference checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from minillm.aira.full_state import AIraQwenCacheUpdater
from minillm.aira.transition_data import TransitionCorpus, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/experiments/aira_qwen_active_v1.json"
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    state_path = Path(config["state_checkpoint"])
    conv_path = Path(config["conv_checkpoint"])
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    conv = torch.load(conv_path, map_location="cpu", weights_only=True)
    state_config = state["config"]
    conv_config = conv["config"]
    layers = list(config["normalization"]["candidate_layers"])
    if layers != list(state_config["candidate_recurrent_layers"]) or layers != list(
        conv_config["candidate_recurrent_layers"]
    ):
        raise ValueError("active and component checkpoints target different layers")
    corpus = TransitionCorpus.load(config["source_pairs"], config["raw_state_work_dir"])
    audit_normalization = corpus.event_normalization(
        layers,
        minimum_standard_deviation=config["normalization"][
            "minimum_standard_deviation"
        ],
    )
    # Both accepted component checkpoints were trained with their internal per-sample
    # LayerNorm on raw event features. Corpus statistics are hash-bound for audit and
    # future training, but applying tiny-sample z-scores regressed held-out state MSE.
    applied_mean = torch.zeros_like(audit_normalization.mean)
    applied_scale = torch.ones_like(audit_normalization.scale)
    model = AIraQwenCacheUpdater(
        event_mean=applied_mean,
        event_scale=applied_scale,
        layers=len(layers),
        state_hidden_dim=state_config["hidden_dim"],
        conv_hidden_dim=conv_config["hidden_dim"],
        conv_bottleneck_dim=conv_config["bottleneck_dim"],
        identity_dim=state_config["identity_dim"],
        state_alpha=1.0,
    )
    model.state_updater.load_state_dict(state["model"])
    model.conv_updater.load_state_dict(conv["model"])
    normalization = {
        "mode": config["normalization"]["mode"],
        "corpus_statistics": config["normalization"]["corpus_statistics"],
        "records": audit_normalization.records,
        "audit_mean_sha256": hashlib.sha256(
            audit_normalization.mean.numpy().tobytes()
        ).hexdigest(),
        "audit_scale_sha256": hashlib.sha256(
            audit_normalization.scale.numpy().tobytes()
        ).hexdigest(),
        "applied_mean_sha256": hashlib.sha256(
            applied_mean.numpy().tobytes()
        ).hexdigest(),
        "applied_scale_sha256": hashlib.sha256(
            applied_scale.numpy().tobytes()
        ).hexdigest(),
    }
    output = Path(args.output or config["combined_checkpoint"])
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "kind": "aira-qwen-normalized-cache-updater",
        "model": model.state_dict(),
        "config": config,
        "config_sha256": sha256(config_path),
        "source_manifest_sha256": sha256(corpus.manifest_path),
        "state_checkpoint_sha256": sha256(state_path),
        "conv_checkpoint_sha256": sha256(conv_path),
        "normalization": normalization,
        "layers": layers,
    }
    torch.save(payload, output)
    report = {
        "schema_version": 1,
        "checkpoint": str(output),
        "checkpoint_sha256": sha256(output),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "state_checkpoint": str(state_path),
        "state_checkpoint_sha256": payload["state_checkpoint_sha256"],
        "conv_checkpoint": str(conv_path),
        "conv_checkpoint_sha256": payload["conv_checkpoint_sha256"],
        "normalization": normalization,
        "source_manifest_sha256": payload["source_manifest_sha256"],
        "stored_answer_routes_used": False,
    }
    report_path = Path(config["status_report"]).with_name(
        "aira_qwen_combined_checkpoint_v1.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
