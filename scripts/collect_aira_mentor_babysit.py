#!/usr/bin/env python3
"""Collect verifier-backed Babysit corrections from a trained AIra Mentor student."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from train_aira_mentor_tiny import prompt_ids

from minillm.aira import (
    BabysitRecord,
    VerifierObservation,
    generate_aira_mentor_records,
    verify_synthetic_generation,
    write_babysit_dataset,
)
from minillm.generation import SamplingConfig, generate_ids, load_model_checkpoint
from minillm.tokenization import load_tokenizer


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def first_difference(first: str, second: str) -> int:
    for index, (left, right) in enumerate(zip(first, second)):
        if left != right:
            return index
    return min(len(first), len(second))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default="artifacts/aira-mentor-tiny-v1/model.pt"
    )
    parser.add_argument(
        "--tokenizer", default="artifacts/tokenizer-github-pilot-v1/tokenizer.json"
    )
    parser.add_argument("--examples-per-category", type=int, default=20)
    parser.add_argument("--task-seed", type=int, default=43)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--output", default="artifacts/aira-mentor-babysit-v1/records.jsonl"
    )
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)
    tokenizer_path = Path(args.tokenizer)
    tokenizer = load_tokenizer(tokenizer_path)
    loaded = load_model_checkpoint(checkpoint_path)
    student_hash = sha256(checkpoint_path)
    records = generate_aira_mentor_records(
        examples_per_category=args.examples_per_category,
        seed=args.task_seed,
    )
    babysit = []
    category_passes: Counter[str] = Counter()
    category_totals: Counter[str] = Counter()
    eos = tokenizer.token_to_id("<eos>")
    assert eos is not None
    for record in records:
        payload = record.to_dict()
        generated = generate_ids(
            loaded.model,
            prompt_ids(payload, tokenizer),
            SamplingConfig(
                max_new_tokens=args.max_new_tokens,
                temperature=0,
                use_cache=True,
            ),
            stop_token_ids={eos},
        )
        student_answer = tokenizer.decode(
            list(generated.generated_token_ids), skip_special_tokens=True
        )
        reference = record.messages[-1].content
        passed = verify_synthetic_generation(payload, student_answer)
        category_totals[record.category] += 1
        category_passes[record.category] += passed
        language = record.language
        if passed:
            critique = ""
            correction = student_answer
            error_type = None
            error_offset = None
            verdict = "correct"
            detail = "Category verifier accepted the generated response."
        else:
            critique = (
                "Ответ не прошёл детерминированную проверку категории; используй проверенный эталон и не копируй неверные числа, ссылки или JSON."
                if language == "ru"
                else "The response failed its deterministic category verifier; use the verified target and do not copy incorrect numbers, citations, or JSON."
            )
            correction = reference
            error_type = f"failed_{record.verification['kind']}"
            error_offset = first_difference(student_answer, reference)
            verdict = "incorrect"
            detail = json.dumps(record.verification, ensure_ascii=False, sort_keys=True)
        prompt = f"SYSTEM:\n{record.messages[0].content}\nUSER:\n{record.messages[1].content}"
        babysit.append(
            BabysitRecord(
                task_id=f"babysit-seed-{args.task_seed}:{record.identifier}",
                prompt=prompt,
                student_checkpoint=student_hash,
                teacher_id="aira-deterministic-reference-v1",
                student_answer=student_answer,
                verifier_observations=(
                    VerifierObservation(
                        tool=f"aira-{record.verification['kind']}",
                        passed=passed,
                        detail=detail,
                    ),
                ),
                verdict=verdict,
                critique=critique,
                corrected_answer=correction,
                rubric_scores={
                    "correctness": 1.0 if passed else 0.0,
                    "format": 1.0 if passed else 0.0,
                    "grounding": 1.0 if passed else 0.0,
                },
                teacher_confidence=1.0,
                first_error_offset=error_offset,
                error_type=error_type,
            )
        )
    output = Path(args.output)
    manifest_path = write_babysit_dataset(
        output,
        babysit,
        metadata={
            "task_generator": "aira-mentor-v1",
            "task_seed": args.task_seed,
            "examples_per_category": args.examples_per_category,
            "student_checkpoint_sha256": student_hash,
            "tokenizer_sha256": sha256(tokenizer_path),
            "protected_v1_splits_used": False,
        },
    )
    report = {
        "schema_version": 1,
        "records": len(babysit),
        "passed": sum(category_passes.values()),
        "failed": len(babysit) - sum(category_passes.values()),
        "category_passes": dict(sorted(category_passes.items())),
        "category_totals": dict(sorted(category_totals.items())),
        "student_checkpoint_sha256": student_hash,
        "task_seed": args.task_seed,
        "manifest": str(manifest_path),
    }
    report_path = output.parent / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
