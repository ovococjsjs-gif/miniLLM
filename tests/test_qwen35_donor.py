from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from minillm.aira.synthetic import SyntheticRecord, generate_aira_mentor_records
from minillm.aira.verification import synthetic_generation_components

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def _assistant_answer(record: SyntheticRecord) -> str:
    return next(message.content for message in record.messages if message.role == "assistant")


def _evaluation_sample(record: SyntheticRecord, generated: str) -> dict[str, object]:
    components = synthetic_generation_components(record, generated)
    return {
        "id": record.identifier,
        "category": record.category,
        "language": record.language,
        "answer": generated,
        "answer_sha256": hashlib.sha256(generated.encode()).hexdigest(),
        "passed": components["strict"],
        "components": components,
    }


def _write_collection_inputs(
    tmp_path: Path,
    records: list[SyntheticRecord],
    generated: list[str],
) -> tuple[Path, Path, Path]:
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        "".join(json.dumps(record.to_dict(), ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(
        json.dumps(
            {
                "samples": [
                    _evaluation_sample(record, answer)
                    for record, answer in zip(records, generated, strict=True)
                ]
            }
        ),
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "checkpoint.gguf"
    checkpoint_path.write_bytes(b"test checkpoint")
    return tasks_path, evaluation_path, checkpoint_path


def test_balanced_selection_is_deterministic_and_category_stratified() -> None:
    evaluator = _load_script("evaluate_qwen35_donor.py")
    records = [
        record.to_dict()
        for record in generate_aira_mentor_records(examples_per_category=8, seed=41)
    ]

    first = evaluator.select_records(records, examples_per_category=2)
    second = evaluator.select_records(records, examples_per_category=2)

    assert [item["id"] for item in first] == [item["id"] for item in second]
    counts: dict[str, int] = {}
    for record in first:
        category = record["category"]
        counts[category] = counts.get(category, 0) + 1
    assert len(counts) == 10
    assert set(counts.values()) == {2}


def test_balanced_selection_rejects_insufficient_categories() -> None:
    evaluator = _load_script("evaluate_qwen35_donor.py")
    records = [
        record.to_dict()
        for record in generate_aira_mentor_records(examples_per_category=2, seed=41)
    ]

    with pytest.raises(ValueError, match="cannot satisfy per-category selection"):
        evaluator.select_records(records, examples_per_category=3)


def test_babysit_collector_recomputes_verification_and_hashes(tmp_path: Path) -> None:
    collector = _load_script("collect_qwen_donor_babysit.py")
    records = generate_aira_mentor_records(examples_per_category=1, seed=52)[:2]
    generated = [_assistant_answer(records[0]), "definitely incorrect"]
    tasks_path, evaluation_path, checkpoint_path = _write_collection_inputs(
        tmp_path, records, generated
    )
    output = tmp_path / "babysit" / "records.jsonl"

    report = collector.collect(
        tasks_path=tasks_path,
        evaluation_path=evaluation_path,
        checkpoint_path=checkpoint_path,
        task_seed=52,
        output=output,
    )

    assert report["records"] == 2
    assert report["passed"] == 1
    assert report["failed"] == 1
    record_lines = output.read_text(encoding="utf-8").splitlines()
    assert len(record_lines) == 2
    correction = json.loads(record_lines[1])
    assert correction["verdict"] == "incorrect"
    assert correction["corrected_answer"] == _assistant_answer(records[1])
    expected_hash = hashlib.sha256(("\n".join(record_lines) + "\n").encode()).hexdigest()
    manifest = json.loads(
        output.with_suffix(".jsonl.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["sha256"] == expected_hash


def test_babysit_collector_rejects_modified_generation_without_recomputed_scores(
    tmp_path: Path,
) -> None:
    collector = _load_script("collect_qwen_donor_babysit.py")
    record = generate_aira_mentor_records(examples_per_category=1, seed=61)[0]
    tasks_path, evaluation_path, checkpoint_path = _write_collection_inputs(
        tmp_path, [record], [_assistant_answer(record)]
    )
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["samples"][0]["answer"] = "tampered after evaluation"
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")

    with pytest.raises(ValueError, match="component verification mismatch"):
        collector.collect(
            tasks_path=tasks_path,
            evaluation_path=evaluation_path,
            checkpoint_path=checkpoint_path,
            task_seed=61,
            output=tmp_path / "out" / "records.jsonl",
        )


def test_babysit_collector_rejects_protected_content_by_default(tmp_path: Path) -> None:
    collector = _load_script("collect_qwen_donor_babysit.py")
    record = generate_aira_mentor_records(examples_per_category=1, seed=42)[0]
    tasks_path, evaluation_path, checkpoint_path = _write_collection_inputs(
        tmp_path, [record], ["wrong"]
    )

    with pytest.raises(ValueError, match="protected AIra Mentor v1 content"):
        collector.collect(
            tasks_path=tasks_path,
            evaluation_path=evaluation_path,
            checkpoint_path=checkpoint_path,
            task_seed=42,
            output=tmp_path / "out" / "records.jsonl",
        )
