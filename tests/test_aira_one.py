from __future__ import annotations

import hashlib
import json
from pathlib import Path

from minillm.aira.babysit import read_babysit_dataset
from minillm.aira.one import (
    AIraBabysitJournal,
    AIraMode,
    AIraOne,
    SkillShelf,
)
from minillm.aira.provider import ProviderResponse
from minillm.aira.synthetic import generate_aira_mentor_records
from minillm.aira.verification import verify_synthetic_generation
from minillm.memory import EpisodicMemoryStore


class FakeProvider:
    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.calls = []

    def complete(
        self,
        messages,
        *,
        temperature=0,
        max_tokens=256,
        extra_body=None,
    ) -> ProviderResponse:
        self.calls.append((messages, temperature, max_tokens, extra_body))
        return ProviderResponse(
            content=self.answers.pop(0),
            reasoning_content="not persisted",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=5,
            raw_model="fake",
        )


def test_exact_routes_pass_fresh_mentor_families_without_neural_model() -> None:
    assistant = AIraOne(None)
    records = generate_aira_mentor_records(examples_per_category=3, seed=57)
    failures = []

    for record in records:
        response = assistant.answer(
            record.messages[1].content,
            system_text=record.messages[0].content,
        )
        if not verify_synthetic_generation(record, response.answer):
            failures.append(
                (record.category, record.identifier, response.route, response.answer)
            )
        assert response.model_bypassed
        assert response.neural_calls == 0

    assert failures == []
    assert assistant.stats.requests == 30
    assert assistant.stats.bypassed_requests == 30
    assert assistant.stats.neural_calls == 0


def test_neural_residual_and_deep_review_are_explicit() -> None:
    provider = FakeProvider(["Draft answer", "Reviewed answer"])
    assistant = AIraOne(provider)

    response = assistant.answer(
        "Explain how a compiler transforms source code.", mode=AIraMode.DEEP
    )

    assert response.answer == "Reviewed answer"
    assert response.route == "neural.residual"
    assert response.model_bypassed is False
    assert response.neural_calls == 2
    assert len(provider.calls) == 2
    assert assistant.stats.neural_calls == 2


def test_external_babysit_skill_shelf_requires_all_topic_groups(tmp_path: Path) -> None:
    path = tmp_path / "skills.json"
    path.write_text(
        json.dumps(
            {
                "skills": [
                    {
                        "skill_id": "science.test",
                        "required_groups": {
                            "ru": [["небо"], ["голуб"]],
                            "en": [["sky"], ["blue"]],
                        },
                        "answers": {
                            "ru": "Проверенный ответ.",
                            "en": "Verified answer.",
                        },
                        "provenance": "test",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    shelf = SkillShelf.load(path)
    assistant = AIraOne(None, skill_shelf=shelf)

    matched = assistant.answer("Почему небо голубое?")
    unmatched = assistant.answer("Почему море голубое?")

    assert matched.answer == "Проверенный ответ."
    assert matched.route == "babysit.broad.science.test"
    assert unmatched.route == "neural.residual"
    assert unmatched.model_bypassed


def test_confirmed_memory_and_babysit_feedback_are_persistent(tmp_path: Path) -> None:
    memory = EpisodicMemoryStore(tmp_path / "memory.sqlite")
    journal_path = tmp_path / "babysit.jsonl"
    journal = AIraBabysitJournal(journal_path)
    assistant = AIraOne(None, memory=memory, journal=journal)

    written = assistant.answer("Запомни: любимый цвет = синий")
    recalled = assistant.answer("Что ты помнишь про любимый цвет?")
    journal.feedback(
        recalled.interaction_id,
        verdict="incorrect",
        correction="Мой любимый цвет — синий.",
    )
    memory.close()

    assert written.route == "memory.write"
    assert "Запомнил" in written.answer
    assert recalled.route == "memory.read"
    assert "синий" in recalled.answer
    lines = [
        json.loads(line)
        for line in journal_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [line["kind"] for line in lines] == [
        "interaction",
        "interaction",
        "feedback",
    ]
    assert lines[-1]["correction"] == "Мой любимый цвет — синий."
    assert all(
        line.get("hidden_reasoning_stored") is False
        for line in lines
        if line["kind"] == "interaction"
    )


def test_aira_one_evidence_records_scope_and_babysit_gain() -> None:
    root = Path(__file__).resolve().parents[1]
    protected = json.loads(
        (root / "results/aira_one_v01_evaluation.json").read_text(encoding="utf-8")
    )
    fresh = json.loads(
        (root / "results/aira_one_v01_fresh_evaluation.json").read_text(
            encoding="utf-8"
        )
    )
    before = json.loads(
        (root / "results/aira_one_v01_prepatch_runtime_smoke.json").read_text(
            encoding="utf-8"
        )
    )
    after = json.loads(
        (root / "results/aira_one_v01_runtime_smoke.json").read_text(encoding="utf-8")
    )
    babysit = read_babysit_dataset(
        root / "artifacts/aira-one-babysit-v01/records.jsonl"
    )

    assert protected["scope"].startswith("deterministic Mentor-family")
    assert protected["strict_passes"] == protected["records"] == 174
    assert protected["neural_calls"] == 0
    assert fresh["strict_passes"] == fresh["records"] == 100
    assert before["neural_calls"] == 4
    assert after["neural_calls"] == 1
    assert before["bypassed_requests"] == 2
    assert after["bypassed_requests"] == 4
    assert len(babysit) == 2
    assert all(record.verdict == "incorrect" for record in babysit)


def test_broad_babysit_evidence_and_deployed_shelf_agree() -> None:
    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (root / "results/aira_one_broad_babysit_v1.json").read_text(encoding="utf-8")
    )
    artifact = root / "artifacts/aira-one-broad-babysit-v1"
    pre_review_skills = json.loads(
        (artifact / "skills_pre_review.json").read_text(encoding="utf-8")
    )
    skills = json.loads((artifact / "skills.json").read_text(encoding="utf-8"))
    audited = json.loads(
        (root / "results/aira_one_broad_babysit_v1_audited.json").read_text(
            encoding="utf-8"
        )
    )
    records = (artifact / "records.jsonl").read_text(encoding="utf-8").splitlines()
    audited_records = (
        (artifact / "records_audited.jsonl").read_text(encoding="utf-8").splitlines()
    )

    assert result["tasks"] == 24
    assert result["verdicts"] == {"correct": 1, "incorrect": 23}
    assert result["validation_passes_before"] == 4
    assert result["validation_passes_after"] == 24
    assert result["validation_neural_calls_before"] == 24
    assert result["validation_neural_calls_after"] == 1
    assert len(pre_review_skills["skills"]) == result["skills_installed"] == 23
    assert len(skills["skills"]) == 24
    assert len(records) == len(read_babysit_dataset(artifact / "records.jsonl")) == 24
    assert (
        len(audited_records)
        == len(read_babysit_dataset(artifact / "records_audited.jsonl"))
        == 25
    )
    audited_manifest = json.loads(
        (artifact / "records_audited.jsonl.manifest.json").read_text(encoding="utf-8")
    )
    assert (
        audited_manifest["sha256"]
        == hashlib.sha256((artifact / "records_audited.jsonl").read_bytes()).hexdigest()
    )
    assert skills["source_records_sha256"] == audited_manifest["sha256"]
    assert [cycle["validation_passes_after"] for cycle in result["cycle_reports"]] == [
        8,
        8,
        8,
    ]
    assert audited["manual_three_cycle"]["validation_passes_before"] == 2
    assert audited["manual_three_cycle"]["validation_passes_after_three_cycles"] == 23
    assert audited["final_independent_regression"]["passes"] == 24
    assert audited["final_independent_regression"]["neural_calls"] == 0
    assert audited["final_independent_regression"]["skills_installed"] == 24
    assert (
        audited["final_independent_regression"][
            "baseline_total_request_latency_seconds"
        ]
        > 249
    )
    assert (
        audited["final_independent_regression"]["total_request_latency_seconds"] < 0.1
    )

    assistant = AIraOne(None)
    boiling = assistant.answer(
        "Если внешнее давление уменьшить, как изменится температура кипения жидкости?"
    )
    assert boiling.route == "babysit.broad.science.boiling-pressure"
    assert boiling.model_bypassed
    assert boiling.neural_calls == 0
    assert "более низкой температуре" in boiling.answer


def test_aira_one_package_manifest_hashes_every_source_file() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "artifacts/aira-one-v01/manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["name"] == "AIra One v0.1"
    assert manifest["experimental_recurrent_bypass_enabled"] is False
    assert len(manifest["files"]) == 32
    for item in manifest["files"]:
        path = root / item["path"]
        assert path.stat().st_size == item["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
