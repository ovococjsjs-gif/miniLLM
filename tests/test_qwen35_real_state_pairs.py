from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/qwen35-real-state-pairs-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_state_pair_arrays_match_manifest_and_prompt_split() -> None:
    manifest = json.loads((ARTIFACT / "manifest.json").read_text(encoding="utf-8"))

    arrays = {}
    for name, metadata in manifest["arrays"].items():
        path = ARTIFACT / metadata["path"]
        assert _sha256(path) == metadata["sha256"]
        value = np.load(path, allow_pickle=False)
        assert list(value.shape) == metadata["shape"]
        assert str(value.dtype) == metadata["dtype"]
        arrays[name] = value

    assert manifest["records"] == 48
    assert manifest["split_records"] == {"train": 32, "validation": 16}
    assert arrays["state_before"].shape == (48, 18, 80)
    assert arrays["state_after"].shape == (48, 18, 80)
    assert not np.array_equal(arrays["state_before"], arrays["state_after"])
    assert arrays["event_features"].shape == (48, 32)
    assert np.all(arrays["emitted_byte_mask"].sum(axis=1) >= 1)
    assert np.allclose(arrays["future_probabilities"].sum(axis=1), 1.0)
    assert np.count_nonzero(arrays["split"] == 0) == 32
    assert np.count_nonzero(arrays["split"] == 1) == 16
    continuity = manifest["exact_cache_continuity"]
    assert continuity == {"comparisons": 864, "state_matches": 864, "conv_matches": 864}


def test_projected_real_state_patcher_beats_copy_but_keeps_full_gate_closed() -> None:
    report = json.loads(
        (ROOT / "results/qwen35_real_state_patcher_proxy.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["dataset_manifest_sha256"] == _sha256(ARTIFACT / "manifest.json")
    assert report["total_steps"] == 300
    assert report["train_records"] == 32
    assert report["validation_records"] == 16
    assert report["validation"]["state_mse_ratio"] < 1.0
    assert report["validation"]["patch_state_mse"] < report["validation_baselines"][
        "mean_train_delta_state_mse"
    ]
    assert report["validation"]["future_kl_improvement"] > 0
    assert report["acceptance"]["beats_copy_state_mse"] is True
    assert report["acceptance"]["beats_mean_delta_state_mse"] is True
    assert report["acceptance"]["full_state_gate_passed"] is False
    assert report["acceptance"]["generated_quality_gate_passed"] is False
    assert report["acceptance"]["acceleration_claim_allowed"] is False
