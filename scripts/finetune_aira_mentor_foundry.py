#!/usr/bin/env python3
"""Fine-tune the tiny interaction smoke on Teacher Foundry corrections."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from train_aira_mentor_tiny import (
    assistant_loss,
    encode_record,
    evaluate,
    load_records,
    make_batch,
    prompt_ids,
)

from minillm.aira import generate_aira_mentor_records, verify_synthetic_generation
from minillm.generation import (
    SamplingConfig,
    generate_ids,
    load_model_checkpoint,
    save_inference_checkpoint,
)
from minillm.tokenization import load_tokenizer


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def generate_demonstrations(
    model,
    tasks,
    tokenizer,
    *,
    eos_id: int,
) -> list[dict[str, Any]]:
    demonstrations = []
    model.eval()
    for record in tasks:
        payload = record.to_dict()
        generated = generate_ids(
            model,
            prompt_ids(payload, tokenizer),
            SamplingConfig(max_new_tokens=96, temperature=0, use_cache=True),
            stop_token_ids={eos_id},
        )
        answer = tokenizer.decode(
            list(generated.generated_token_ids), skip_special_tokens=True
        )
        demonstrations.append(
            {
                "id": record.identifier,
                "category": record.category,
                "language": record.language,
                "generated": answer,
                "verified_pass": verify_synthetic_generation(payload, answer),
            }
        )
    return demonstrations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initial-checkpoint", default="artifacts/aira-mentor-tiny-v1/model.pt"
    )
    parser.add_argument(
        "--curriculum",
        default="artifacts/aira-teacher-foundry-v1/curriculum.jsonl",
    )
    parser.add_argument(
        "--validation", default="artifacts/aira-mentor-v1/validation.jsonl"
    )
    parser.add_argument(
        "--tokenizer", default="artifacts/tokenizer-github-pilot-v1/tokenizer.json"
    )
    parser.add_argument("--steps", type=int, default=300, choices=range(1, 301))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--fresh-task-seed", type=int, default=45)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--checkpoint", default="artifacts/aira-mentor-tiny-foundry-v1/model.pt"
    )
    parser.add_argument(
        "--output", default="results/aira_mentor_tiny_foundry_finetune.json"
    )
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    initial_checkpoint = Path(args.initial_checkpoint)
    curriculum_path = Path(args.curriculum)
    validation_path = Path(args.validation)
    tokenizer_path = Path(args.tokenizer)
    loaded = load_model_checkpoint(initial_checkpoint)
    model = loaded.model
    tokenizer = load_tokenizer(tokenizer_path)
    train_records = load_records(curriculum_path)
    validation_records = load_records(validation_path)
    train_examples = [
        encode_record(record, tokenizer, maximum_length=loaded.config.max_seq_len)
        for record in train_records
    ]
    validation_examples = [
        encode_record(record, tokenizer, maximum_length=loaded.config.max_seq_len)
        for record in validation_records
    ]
    pad_id = tokenizer.token_to_id("<pad>")
    eos_id = tokenizer.token_to_id("<eos>")
    assert pad_id is not None and eos_id is not None
    fresh_tasks = generate_aira_mentor_records(
        examples_per_category=1, seed=args.fresh_task_seed
    )
    initial_demonstrations = generate_demonstrations(
        model, fresh_tasks, tokenizer, eos_id=eos_id
    )
    initial_validation = evaluate(
        model,
        validation_examples,
        validation_records,
        batch_size=16,
        pad_id=pad_id,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    samples = []
    started = time.perf_counter()
    model.train()
    for step in range(args.steps):
        indices = [rng.randrange(len(train_examples)) for _ in range(args.batch_size)]
        input_ids, labels = make_batch(train_examples, indices, pad_id)
        loss = assistant_loss(model, input_ids, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite Foundry loss at step {step + 1}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in {0, args.steps // 4, args.steps // 2, 3 * args.steps // 4, args.steps - 1}:
            samples.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "gradient_norm": float(gradient_norm),
                }
            )
    training_seconds = time.perf_counter() - started
    final_validation = evaluate(
        model,
        validation_examples,
        validation_records,
        batch_size=16,
        pad_id=pad_id,
    )

    final_demonstrations = generate_demonstrations(
        model, fresh_tasks, tokenizer, eos_id=eos_id
    )

    checkpoint = Path(args.checkpoint)
    save_inference_checkpoint(
        checkpoint,
        model,
        step=args.steps,
        metadata={
            "experiment": "aira-mentor-tiny-foundry-finetune-v1",
            "parent_checkpoint_sha256": sha256(initial_checkpoint),
            "curriculum_sha256": sha256(curriculum_path),
            "optimizer_steps_this_run": args.steps,
            "seed": args.seed,
        },
    )
    initial_passes = sum(
        item["verified_pass"] for item in initial_demonstrations
    )
    final_passes = sum(item["verified_pass"] for item in final_demonstrations)
    report = {
        "schema_version": 1,
        "experiment": "aira-mentor-tiny-foundry-finetune-v1",
        "warning": (
            "The parent remains a 1.7M random-init interaction smoke. This run tests "
            "the Foundry path; it does not create a general pretrained assistant."
        ),
        "parent_checkpoint": str(initial_checkpoint),
        "parent_checkpoint_sha256": sha256(initial_checkpoint),
        "curriculum": str(curriculum_path),
        "curriculum_sha256": sha256(curriculum_path),
        "curriculum_records": len(train_records),
        "validation": str(validation_path),
        "tokenizer_sha256": sha256(tokenizer_path),
        "seed": args.seed,
        "fresh_task_seed": args.fresh_task_seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_seconds": training_seconds,
        "loss_samples": samples,
        "initial_validation": initial_validation,
        "final_validation": final_validation,
        "initial_fresh_demonstration_passes": initial_passes,
        "final_fresh_demonstration_passes": final_passes,
        "fresh_demonstration_count": len(final_demonstrations),
        "initial_fresh_demonstrations": initial_demonstrations,
        "final_fresh_demonstrations": final_demonstrations,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
