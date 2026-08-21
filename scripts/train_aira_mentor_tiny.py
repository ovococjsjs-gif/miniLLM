#!/usr/bin/env python3
"""Train a tiny random-init assistant-only AIra Mentor interaction smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from minillm.config import MiniLLMConfig
from minillm.generation import SamplingConfig, generate_ids, save_inference_checkpoint
from minillm.model import MiniLLM
from minillm.tokenization import load_tokenizer


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def encode_record(
    record: dict[str, Any], tokenizer, *, maximum_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    roles = {message["role"]: message["content"] for message in record["messages"]}
    required = ("system", "user", "assistant")
    if any(role not in roles for role in required):
        raise ValueError("AIra Mentor record lacks system/user/assistant messages")
    special = {
        name: tokenizer.token_to_id(name)
        for name in ("<bos>", "<eos>", "<system>", "<user>", "<assistant>")
    }
    if any(value is None for value in special.values()):
        raise ValueError("tokenizer lacks required chat special tokens")
    prefix = [special["<bos>"], special["<system>"]]
    prefix += tokenizer.encode(roles["system"]).ids
    prefix += [special["<user>"]]
    prefix += tokenizer.encode(roles["user"]).ids
    prefix += [special["<assistant>"]]
    answer = tokenizer.encode(roles["assistant"]).ids + [special["<eos>"]]
    if len(prefix) + len(answer) > maximum_length:
        available = maximum_length - len(prefix)
        if available < 2:
            raise ValueError("chat prefix exceeds model context")
        answer = answer[:available]
        answer[-1] = special["<eos>"]
    ids = torch.tensor(prefix + answer, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    labels[len(prefix) :] = ids[len(prefix) :]
    return ids, labels


def make_batch(
    examples: list[tuple[torch.Tensor, torch.Tensor]], indices: list[int], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = [examples[index] for index in indices]
    length = max(len(ids) for ids, _ in selected)
    input_ids = torch.full((len(selected), length), pad_id, dtype=torch.long)
    labels = torch.full((len(selected), length), -100, dtype=torch.long)
    for row, (ids, target) in enumerate(selected):
        input_ids[row, : len(ids)] = ids
        labels[row, : len(target)] = target
    return input_ids, labels


def assistant_loss(
    model: MiniLLM, input_ids: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    output = model(input_ids)
    return F.cross_entropy(
        output.logits[:, :-1].reshape(-1, model.config.vocab_size),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


def evaluate(
    model: MiniLLM,
    examples: list[tuple[torch.Tensor, torch.Tensor]],
    records: list[dict[str, Any]],
    *,
    batch_size: int,
    pad_id: int,
) -> dict[str, Any]:
    totals: dict[str, list[float]] = defaultdict(list)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            indices = list(range(start, min(start + batch_size, len(examples))))
            input_ids, labels = make_batch(examples, indices, pad_id)
            logits = model(input_ids).logits
            token_losses = F.cross_entropy(
                logits[:, :-1].reshape(-1, model.config.vocab_size),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).reshape(len(indices), -1)
            valid = labels[:, 1:] != -100
            for row, index in enumerate(indices):
                loss = float(token_losses[row][valid[row]].mean())
                totals[records[index]["category"]].append(loss)
                totals["overall"].append(loss)
    return {
        category: {
            "records": len(losses),
            "nll": sum(losses) / len(losses),
            "perplexity": math.exp(sum(losses) / len(losses)),
        }
        for category, losses in sorted(totals.items())
    }


def prompt_ids(record: dict[str, Any], tokenizer) -> list[int]:
    roles = {message["role"]: message["content"] for message in record["messages"]}
    ids = [tokenizer.token_to_id("<bos>"), tokenizer.token_to_id("<system>")]
    ids += tokenizer.encode(roles["system"]).ids
    ids += [tokenizer.token_to_id("<user>")]
    ids += tokenizer.encode(roles["user"]).ids
    ids += [tokenizer.token_to_id("<assistant>")]
    return [int(value) for value in ids]


def verify_generation(record: dict[str, Any], generated: str) -> bool:
    verification = record["verification"]
    expected = verification.get("expected")
    category = record["category"]
    if verification["kind"] == "json_equal":
        try:
            return json.loads(generated) == expected
        except json.JSONDecodeError:
            return False
    if verification["kind"] == "python_tests":
        match = re.search(r"```python\n(.*?)\n```", generated, flags=re.DOTALL)
        if match is None or verification["function"] not in match.group(1):
            return False
        try:
            compile(match.group(1), "<generated>", "exec")
        except SyntaxError:
            return False
        return True
    if category == "arithmetic":
        return (
            re.search(rf"(?:Answer:|Ответ:)\s*{expected}\.?\s*$", generated) is not None
        )
    if category == "algebra":
        return re.search(rf"[xyz]\s*=\s*{expected}(?:\D|$)", generated) is not None
    if category == "logic":
        reference = next(
            message["content"]
            for message in record["messages"]
            if message["role"] == "assistant"
        )
        return generated == reference
    if category == "memory_control":
        reference = next(
            message["content"]
            for message in record["messages"]
            if message["role"] == "assistant"
        )
        return generated == reference
    if category in {"grounded_qa", "prompt_injection"}:
        return (
            str(expected) in generated and f"[{verification['citation']}]" in generated
        )
    if category == "uncertainty":
        reference = next(
            message["content"]
            for message in record["messages"]
            if message["role"] == "assistant"
        )
        return generated == reference
    if category == "critique_revision":
        return generated.rstrip(".").endswith(str(expected))
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="artifacts/aira-mentor-v1")
    parser.add_argument("--model", default="configs/aira_mentor_tiny.json")
    parser.add_argument(
        "--tokenizer", default="artifacts/tokenizer-github-pilot-v1/tokenizer.json"
    )
    parser.add_argument("--steps", type=int, default=300, choices=range(1, 301))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--checkpoint", default="artifacts/aira-mentor-tiny-v1/model.pt"
    )
    parser.add_argument("--output", default="results/aira_mentor_tiny_training.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dataset = Path(args.dataset)
    model_path = Path(args.model)
    tokenizer_path = Path(args.tokenizer)
    config = MiniLLMConfig.load(model_path)
    tokenizer = load_tokenizer(tokenizer_path)
    train_records = load_records(dataset / "train.jsonl")
    validation_records = load_records(dataset / "validation.jsonl")
    train_examples = [
        encode_record(record, tokenizer, maximum_length=config.max_seq_len)
        for record in train_records
    ]
    validation_examples = [
        encode_record(record, tokenizer, maximum_length=config.max_seq_len)
        for record in validation_records
    ]
    pad_id = tokenizer.token_to_id("<pad>")
    assert pad_id is not None
    model = MiniLLM(config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    initial_validation = evaluate(
        model,
        validation_examples,
        validation_records,
        batch_size=16,
        pad_id=pad_id,
    )
    samples = []
    started = time.perf_counter()
    model.train()
    for step in range(args.steps):
        indices = [rng.randrange(len(train_examples)) for _ in range(args.batch_size)]
        input_ids, labels = make_batch(train_examples, indices, pad_id)
        loss = assistant_loss(model, input_ids, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite SFT loss at step {step + 1}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in {
            0,
            args.steps // 4,
            args.steps // 2,
            3 * args.steps // 4,
            args.steps - 1,
        }:
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
    demonstrations = []
    seen_categories = set()
    for record in validation_records:
        if record["category"] in seen_categories:
            continue
        seen_categories.add(record["category"])
        generated = generate_ids(
            model,
            prompt_ids(record, tokenizer),
            SamplingConfig(max_new_tokens=48, temperature=0, use_cache=True),
            stop_token_ids={int(tokenizer.token_to_id("<eos>"))},
        )
        generated_text = tokenizer.decode(
            list(generated.generated_token_ids), skip_special_tokens=True
        )
        demonstrations.append(
            {
                "id": record["id"],
                "category": record["category"],
                "language": record["language"],
                "generated": generated_text,
                "verified_pass": verify_generation(record, generated_text),
                "reference": next(
                    message["content"]
                    for message in record["messages"]
                    if message["role"] == "assistant"
                ),
            }
        )
    checkpoint = Path(args.checkpoint)
    save_inference_checkpoint(
        checkpoint,
        model,
        step=args.steps,
        metadata={
            "purpose": "random-init AIra Mentor interaction smoke",
            "dataset_manifest_sha256": sha256(dataset / "manifest.json"),
            "tokenizer_sha256": sha256(tokenizer_path),
            "model_config_sha256": sha256(model_path),
            "seed": args.seed,
        },
    )
    report = {
        "schema_version": 1,
        "experiment": "aira-mentor-tiny-random-init-sft-v1",
        "warning": "Random-init 300-step interaction smoke on 0.7M synthetic SFT tokens; not a pretrained base or useful general assistant.",
        "environment": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "threads": args.threads,
        },
        "dataset": str(dataset),
        "dataset_manifest_sha256": sha256(dataset / "manifest.json"),
        "model_config": str(model_path),
        "model_config_sha256": sha256(model_path),
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": sha256(tokenizer_path),
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_seconds": training_seconds,
        "training_examples_per_second": args.steps * args.batch_size / training_seconds,
        "loss_samples": samples,
        "initial_validation": initial_validation,
        "final_validation": final_validation,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "demonstration_verified_passes": sum(
            demonstration["verified_pass"] for demonstration in demonstrations
        ),
        "demonstration_count": len(demonstrations),
        "demonstrations": demonstrations,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
