"""Multi-query associative recall (MQAR) probe.

This generated task isolates whether a model can bind keys to values and retrieve the
right values later. Instances are created from a seed, so memorizing a static benchmark
cannot improve held-out accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RecallBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    prediction_positions: torch.Tensor
    answers: torch.Tensor


def generate_recall_batch(
    batch_size: int,
    *,
    pairs: int = 4,
    queries: int = 2,
    key_count: int = 32,
    value_count: int = 32,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> RecallBatch:
    if queries > pairs:
        raise ValueError("queries cannot exceed pairs")
    # 0=PAD, 1=BOS, 2=SEP, 3=EOS; keys and values use disjoint ranges.
    key_offset = 4
    value_offset = key_offset + key_count
    rows: list[list[int]] = []
    answer_rows: list[list[int]] = []
    for _ in range(batch_size):
        key_perm = torch.randperm(key_count, generator=generator)[:pairs].tolist()
        value_ids = torch.randint(value_count, (pairs,), generator=generator).tolist()
        query_slots = torch.randperm(pairs, generator=generator)[:queries].tolist()
        row = [1]
        for key, value in zip(key_perm, value_ids):
            row.extend((key_offset + key, value_offset + value))
        row.append(2)
        answers: list[int] = []
        for slot in query_slots:
            row.extend((key_offset + key_perm[slot], value_offset + value_ids[slot]))
            answers.append(value_offset + value_ids[slot])
        row.append(3)
        rows.append(row)
        answer_rows.append(answers)

    input_ids = torch.tensor(rows, dtype=torch.long, device=device)
    labels = torch.full_like(input_ids, -100)
    # Layout: BOS, 2*pairs bindings, SEP, then alternating query/answer.
    first_answer = 2 * pairs + 3
    answer_positions = torch.arange(
        first_answer, first_answer + 2 * queries, 2, device=device
    )
    labels[:, answer_positions] = input_ids[:, answer_positions]
    prediction_positions = answer_positions - 1
    answers = torch.tensor(answer_rows, dtype=torch.long, device=device)
    return RecallBatch(input_ids, labels, prediction_positions, answers)


@torch.no_grad()
def evaluate_recall(model: torch.nn.Module, batch: RecallBatch) -> float:
    model.eval()
    logits = model(batch.input_ids).logits
    batch_indices = torch.arange(batch.input_ids.shape[0], device=logits.device)[
        :, None
    ]
    predictions = logits[batch_indices, batch.prediction_positions[None, :]].argmax(
        dim=-1
    )
    return float((predictions == batch.answers).float().mean())
