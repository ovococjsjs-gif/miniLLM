from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/aira-qwen-active-v1"


def test_active_config_leaves_only_bounded_training_and_gates() -> None:
    config = json.loads(
        (ROOT / "configs/experiments/aira_qwen_active_v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert config["training"]["state_steps"] == 300
    assert config["training"]["conv_steps"] == 300
    assert config["training"]["per_run_step_cap"] == 300
    assert config["training"]["stored_answer_routes"] is False
    assert config["normalization"]["mode"] == "per-sample-layer-norm"
    assert config["normalization"]["corpus_statistics"] == "audit-only"
    assert config["evaluation"]["transitions"] == 16
    assert config["deployment_allowed"] is False


def test_combined_checkpoint_binds_components_and_normalization() -> None:
    checkpoint_path = ARTIFACT / "cache-updater.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    report = json.loads(
        (ROOT / "results/aira_qwen_combined_checkpoint_v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert checkpoint["schema_version"] == 2
    assert checkpoint["kind"] == "aira-qwen-normalized-cache-updater"
    assert checkpoint["layers"] == [4, 8, 12, 16, 20]
    assert checkpoint["normalization"]["records"] == 160
    assert report["parameters"] == 8_069_185
    assert report["stored_answer_routes_used"] is False
    assert (
        report["checkpoint_sha256"]
        == hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    )
    assert report["state_checkpoint_sha256"] == checkpoint["state_checkpoint_sha256"]
    assert report["conv_checkpoint_sha256"] == checkpoint["conv_checkpoint_sha256"]
    assert torch.equal(checkpoint["model"]["event_mean"], torch.zeros(1056))
    assert torch.equal(checkpoint["model"]["event_scale"], torch.ones(1056))

    state_component = torch.load(
        ROOT / "artifacts/qwen35-gated-delta-updater-v1/model.pt",
        map_location="cpu",
        weights_only=True,
    )["model"]
    conv_component = torch.load(
        ROOT / "artifacts/qwen35-conv-row-updater-v1/model.pt",
        map_location="cpu",
        weights_only=True,
    )["model"]
    for key, value in state_component.items():
        assert torch.equal(checkpoint["model"][f"state_updater.{key}"], value)
    for key, value in conv_component.items():
        assert torch.equal(checkpoint["model"][f"conv_updater.{key}"], value)


def test_active_pipeline_runs_through_strict_full_cache_replay() -> None:
    report = json.loads(
        (ROOT / "results/aira_qwen_active_v1.json").read_text(encoding="utf-8")
    )

    assert report["mode"] == "trained-and-replayed"
    assert report["prepared"]["ready_for_training"]
    assert report["prepared"]["stored_answer_routes_used"] is False
    assert report["prepared"]["normalization_records"] == 160
    assert report["calibration"]["split"] == "train-only"
    assert report["calibration"]["metric"] == "full-cache"
    assert report["calibration"]["validation_prompts_used"] == 0
    assert report["replay"]["transitions"] == 16
    assert report["replay"]["learned_full_over_copy_kl_ratio"] < 0.84
    assert report["replay"]["learned_full_argmax_preserved"] == 16
    assert report["gates"]["normalization_contract_hash_bound"]
    assert report["gates"]["combined_checkpoint"]
    assert report["gates"]["full_cache_mean_kl"]
    assert not report["gates"]["every_transition_kl"]
    assert not report["gates"]["deployment_allowed"]


def test_active_artifact_manifest_is_complete() -> None:
    manifest = json.loads((ARTIFACT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_ready"]
    assert not manifest["deployment_allowed"]
    for item in manifest["files"]:
        path = ARTIFACT / item["path"]
        assert path.stat().st_size == item["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
