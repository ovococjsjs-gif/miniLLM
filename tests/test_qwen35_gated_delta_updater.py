from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_full_state_updater_beats_copy_on_disjoint_prompt_groups() -> None:
    result = json.loads(
        (ROOT / "results/qwen35_gated_delta_updater_v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert result["steps"] == 300
    assert result["parameters"] == 5_647_184
    assert result["candidate_recurrent_layers"] == [4, 8, 12, 16, 20]
    assert result["structure"]["exact_formula_max_abs"] < 1e-5
    assert (
        result["structure"]["conv_shift_matches"]
        == result["structure"]["conv_shift_comparisons"]
    )
    assert result["validation"]["state_mse_ratio"] < 0.85
    assert result["acceptance"]["beats_copy_state_mse"]
    assert not result["acceptance"]["deployment_allowed"]


def test_patch_alpha_is_calibrated_without_validation_prompts() -> None:
    calibration = json.loads(
        (ROOT / "results/qwen35_learned_state_alpha_calibration_v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert calibration["split"] == "train-only"
    assert calibration["validation_prompts_used"] == 0
    assert calibration["selected_alpha"] == 0.01
    assert calibration["selected_ratio"] < 0.65
    assert all(candidate["prompts"] == 8 for candidate in calibration["candidates"])


def test_native_learned_state_injection_improves_mean_future_kl() -> None:
    replay = json.loads(
        (ROOT / "results/qwen35_learned_state_replay_v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert replay["prompts"] == 4
    assert replay["train_calibrated_alpha"] == 0.01
    assert replay["learned_over_copy_kl_ratio"] < 0.70
    assert replay["learned_improvements"] == 3
    assert replay["argmax_preserved"] == 4
    assert replay["acceptance"]["all_serialization_controls_exact"]
    assert replay["acceptance"]["learned_mean_kl_below_candidate_copy"]
    assert not replay["acceptance"]["learned_improves_every_prompt"]
    assert not replay["acceptance"]["deployment_allowed"]


def test_updater_checkpoint_and_native_sources_are_hash_bound() -> None:
    artifact = ROOT / "artifacts/qwen35-gated-delta-updater-v1"
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = artifact / item["path"]
        assert path.stat().st_size == item["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]

    for result_name, source_name in (
        ("qwen35_transition_probe_build.json", "qwen35_transition_probe.cpp"),
        ("qwen35_learned_state_replay_build.json", "qwen35_learned_state_replay.cpp"),
    ):
        build = json.loads((ROOT / "results" / result_name).read_text(encoding="utf-8"))
        source = ROOT / "native" / source_name
        assert build["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
