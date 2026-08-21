from __future__ import annotations

import hashlib
import json
from pathlib import Path

from minillm.aira.one import AIraBabysitJournal, AIraMode, AIraOne
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
            failures.append((record.category, record.identifier, response.route, response.answer))
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
    lines = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
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


def test_aira_one_evidence_records_only_exact_controller_scope() -> None:
    root = Path(__file__).resolve().parents[1]
    protected = json.loads(
        (root / "results/aira_one_v01_evaluation.json").read_text(encoding="utf-8")
    )
    fresh = json.loads(
        (root / "results/aira_one_v01_fresh_evaluation.json").read_text(
            encoding="utf-8"
        )
    )

    assert protected["scope"].startswith("deterministic Mentor-family")
    assert protected["strict_passes"] == protected["records"] == 174
    assert protected["neural_calls"] == 0
    assert fresh["strict_passes"] == fresh["records"] == 100


def test_feedback_does_not_install_a_stored_answer_route(tmp_path: Path) -> None:
    provider = FakeProvider(["A neural answer", "A second neural answer"])
    journal = AIraBabysitJournal(tmp_path / "feedback.jsonl")
    assistant = AIraOne(provider, journal=journal)

    first = assistant.answer("Почему небо голубое?")
    journal.feedback(
        first.interaction_id,
        verdict="incorrect",
        correction="Исправление для последующего обучения параметров.",
    )
    second = assistant.answer("Почему небо голубое?")

    assert first.route == second.route == "neural.residual"
    assert first.answer == "A neural answer"
    assert second.answer == "A second neural answer"
    assert first.neural_calls == second.neural_calls == 1


def test_aira_one_package_manifest_hashes_every_source_file() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "artifacts/aira-one-v01/manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["name"] == "AIra One v0.1"
    assert manifest["experimental_recurrent_bypass_enabled"] is False
    assert len(manifest["files"]) == 12
    for item in manifest["files"]:
        path = root / item["path"]
        assert path.stat().st_size == item["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
