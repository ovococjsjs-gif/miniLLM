"""Small, auditable completion suites for checkpoint smoke evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from .generation import SamplingConfig, generate_ids
from .model import MiniLLM


@dataclass(frozen=True)
class CompletionCase:
    id: str
    language: str
    category: str
    prompt: str
    max_new_tokens: int = 32
    contains_any: tuple[str, ...] = ()
    forbids_any: tuple[str, ...] = ()
    minimum_characters: int = 1

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CompletionCase:
        data = dict(raw)
        data["contains_any"] = tuple(data.get("contains_any", ()))
        data["forbids_any"] = tuple(data.get("forbids_any", ()))
        case = cls(**data)
        if not case.id or not case.prompt or case.max_new_tokens < 1:
            raise ValueError("completion case has invalid identity, prompt, or length")
        if case.minimum_characters < 0:
            raise ValueError("minimum_characters cannot be negative")
        return case


@dataclass(frozen=True)
class CompletionRecord:
    id: str
    language: str
    category: str
    prompt: str
    completion: str
    generated_token_ids: tuple[int, ...]
    stop_reason: str
    used_cache: bool
    checks: dict[str, bool]
    passed: bool
    error: str | None = None


def load_completion_cases(path: str | Path) -> tuple[CompletionCase, ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("completion suite must be a non-empty JSON list")
    cases = tuple(CompletionCase.from_dict(item) for item in raw)
    identifiers = [case.id for case in cases]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("completion case IDs must be unique")
    return cases


def check_completion(case: CompletionCase, completion: str) -> dict[str, bool]:
    folded = completion.casefold()
    checks = {"minimum_characters": len(completion.strip()) >= case.minimum_characters}
    if case.contains_any:
        checks["contains_any"] = any(
            item.casefold() in folded for item in case.contains_any
        )
    if case.forbids_any:
        checks["forbids_any"] = not any(
            item.casefold() in folded for item in case.forbids_any
        )
    return checks


def evaluate_completion_suite(
    model: MiniLLM,
    tokenizer: Tokenizer,
    cases: tuple[CompletionCase, ...],
    sampling: SamplingConfig | None = None,
    *,
    add_bos: bool = True,
    core_repetitions: int | None = None,
) -> dict[str, Any]:
    """Generate and score transparent smoke checks, not a general quality metric."""

    base_sampling = (sampling or SamplingConfig(temperature=0)).validate()
    bos_id = tokenizer.token_to_id("<bos>") if add_bos else None
    eos_id = tokenizer.token_to_id("<eos>")
    stop_tokens = {eos_id} if eos_id is not None else set()
    records: list[CompletionRecord] = []
    for index, case in enumerate(cases):
        prompt_ids = tokenizer.encode(case.prompt).ids
        if bos_id is not None:
            prompt_ids.insert(0, bos_id)
        if len(prompt_ids) > model.config.max_seq_len:
            records.append(
                CompletionRecord(
                    id=case.id,
                    language=case.language,
                    category=case.category,
                    prompt=case.prompt,
                    completion="",
                    generated_token_ids=(),
                    stop_reason="error",
                    used_cache=False,
                    checks={"generation": False},
                    passed=False,
                    error=(
                        f"prompt has {len(prompt_ids)} tokens but model limit is "
                        f"{model.config.max_seq_len}"
                    ),
                )
            )
            continue
        case_sampling = replace(
            base_sampling,
            max_new_tokens=case.max_new_tokens,
            seed=base_sampling.seed + index,
        )
        result = generate_ids(
            model,
            prompt_ids,
            case_sampling,
            stop_token_ids=stop_tokens,
            core_repetitions=core_repetitions,
        )
        completion = tokenizer.decode(
            list(result.generated_token_ids), skip_special_tokens=True
        )
        checks = check_completion(case, completion)
        records.append(
            CompletionRecord(
                id=case.id,
                language=case.language,
                category=case.category,
                prompt=case.prompt,
                completion=completion,
                generated_token_ids=result.generated_token_ids,
                stop_reason=result.stop_reason,
                used_cache=result.used_cache,
                checks=checks,
                passed=all(checks.values()),
            )
        )
    return {
        "cases": [asdict(record) for record in records],
        "passed": sum(record.passed for record in records),
        "total": len(records),
        "pass_rate": sum(record.passed for record in records) / max(1, len(records)),
        "note": "Smoke checks are diagnostic and must not be treated as aggregate model quality.",
    }
