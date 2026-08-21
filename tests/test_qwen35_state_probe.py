from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_audit_module():
    path = ROOT / "scripts/audit_qwen35_state_probe.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def test_vector_change_reports_exact_delta_and_cosine() -> None:
    module = _load_audit_module()

    result = module.vector_change(array("f", [3.0, 4.0]), array("f", [0.0, 5.0]))

    assert result["before_l2"] == 5.0
    assert result["after_l2"] == 5.0
    assert result["delta_l2"] == result["delta_over_after"] * 5.0
    assert abs(result["cosine"] - 0.8) < 1e-7


def test_real_state_probe_evidence_keeps_acceleration_gate_closed() -> None:
    report = json.loads(
        (ROOT / "results/qwen35_08b_real_state_probe.json").read_text(
            encoding="utf-8"
        )
    )
    build = json.loads(
        (ROOT / "results/qwen35_state_probe_build.json").read_text(encoding="utf-8")
    )
    source = ROOT / "native/qwen35_state_probe.cpp"

    assert hashlib.sha256(source.read_bytes()).hexdigest() == build["source_sha256"]
    assert report["runtime"]["probe_source_sha256"] == build["source_sha256"]
    assert report["architecture"]["recurrent_layers"] == [
        0,
        1,
        2,
        4,
        5,
        6,
        8,
        9,
        10,
        12,
        13,
        14,
        16,
        17,
        18,
        20,
        21,
        22,
    ]
    assert report["captures"]["total_tensor_captures"] == 192
    assert report["cache_continuity"]["recurrent_state_exact_matches"] == 18
    assert report["cache_continuity"]["convolution_state_exact_matches"] == 18
    assert report["cache_continuity"]["recurrent_state_exact"] is True
    assert report["cache_continuity"]["convolution_state_exact"] is True
    gate = report["gate_interpretation"]
    assert gate["real_recurrent_states_captured"] is True
    assert gate["full_logits_captured"] is True
    assert gate["state_patcher_trained_on_real_states"] is False
    assert gate["acceleration_claim_allowed"] is False
