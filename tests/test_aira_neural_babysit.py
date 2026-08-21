from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/aira-neural-babysit-v1"


def test_neural_babysit_changes_parameters_and_improves_held_out_generation() -> None:
    raw = json.loads(
        (ROOT / "results/aira_neural_babysit_v1.json").read_text(encoding="utf-8")
    )
    audited = json.loads(
        (ROOT / "results/aira_neural_babysit_v1_audited.json").read_text(
            encoding="utf-8"
        )
    )

    assert raw["training"]["steps"] == 300
    assert raw["training"]["records"] == 48
    assert raw["training"]["parameters_changed"] == 264_137
    assert raw["free_generation"]["adapted_routes_from_shelf"] == 0
    assert raw["training"]["validation_teacher_forced"]["adapted_nll"] < 0.1
    assert raw["training"]["validation_teacher_forced"]["adapted_top1_accuracy"] > 0.97
    assert audited["manual_quality"]["base_passes"] == 1
    assert audited["manual_quality"]["adapted_passes"] == 13
    assert audited["manual_quality"]["absolute_improvement"] == 12
    assert audited["decision"]["parameter_learning_demonstrated"]
    assert not audited["decision"]["production_deployment_allowed"]


def test_adapter_is_numeric_parameters_not_an_answer_shelf() -> None:
    adapter = (ARTIFACT / "adapter.bin").read_bytes()
    assert adapter[:8] == b"AIRAODA2"
    embedding, hidden, candidates, gate_hidden, gain, threshold = struct.unpack(
        "<IIIIff", adapter[8:32]
    )
    assert (embedding, hidden, candidates, gate_hidden) == (1024, 128, 776, 32)
    assert gain == 1.0
    assert threshold == 0.5
    expected_bytes = (
        32
        + candidates * 4
        + embedding * 4 * 2
        + hidden * embedding * 4
        + hidden * 4
        + candidates * hidden * 4
        + candidates * 4
        + gate_hidden * embedding * 4
        + gate_hidden * 4
        + gate_hidden * 4
        + 4
    )
    assert len(adapter) == expected_bytes

    curriculum = json.loads(
        (ROOT / "configs/aira-one/broad_curriculum_v1.json").read_text(encoding="utf-8")
    )
    for cycle in curriculum["cycles"]:
        for task in cycle["tasks"]:
            answer = task["answers"][task["language"]].encode("utf-8")
            assert answer not in adapter


def test_scrubbed_inference_proves_teacher_answers_are_not_runtime_inputs() -> None:
    report = json.loads(
        (ROOT / "results/aira_neural_adapter_independence_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["teacher_answer_bytes_available_to_runtime"] == 0
    assert report["records"] == 24
    assert report["byte_exact_matches"] == 24
    assert report["all_outputs_byte_exact"]
    for line in (
        (ARTIFACT / "validation_inference.tsv").read_text(encoding="utf-8").splitlines()
    ):
        _, _, answer_hex = line.split("\t")
        assert bytes.fromhex(answer_hex) == b"-"


def test_neural_train_and_held_out_prompts_are_disjoint() -> None:
    train_prompts = {
        bytes.fromhex(line.split("\t")[1]).decode("utf-8")
        for line in (ARTIFACT / "train.tsv").read_text(encoding="utf-8").splitlines()
    }
    validation_prompts = {
        bytes.fromhex(line.split("\t")[1]).decode("utf-8")
        for line in (ARTIFACT / "validation.tsv")
        .read_text(encoding="utf-8")
        .splitlines()
    }
    assert len(train_prompts) == 48
    assert len(validation_prompts) == 24
    assert train_prompts.isdisjoint(validation_prompts)


def test_neural_babysit_manifest_hashes_all_evidence() -> None:
    manifest = json.loads((ARTIFACT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_steps"] == 300
    assert manifest["stored_answer_routes_used"] is False
    for item in manifest["files"]:
        path = ARTIFACT / item["path"]
        assert path.stat().st_size == item["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
