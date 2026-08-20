"""Generated permutation pointer-chasing task for iterative-depth experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PointerChaseBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    prediction_positions: torch.Tensor
    answers: torch.Tensor
    hops: torch.Tensor


def generate_pointer_chase_batch(
    batch_size: int,
    *,
    node_count: int = 8,
    min_hops: int = 1,
    max_hops: int = 4,
    fixed_hops: int | None = None,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> PointerChaseBatch:
    """Encode one random full cycle and ask for its h-fold application to a node."""

    if node_count < 2:
        raise ValueError("node_count must be at least two")
    if not 1 <= min_hops <= max_hops:
        raise ValueError("invalid hop range")
    if fixed_hops is not None and not min_hops <= fixed_hops <= max_hops:
        raise ValueError("fixed_hops is outside the configured range")
    # 0=PAD, 1=BOS, 2=SEP, 3=EOS, then nodes and hop-count control tokens.
    node_offset = 4
    hop_offset = node_offset + node_count
    rows = []
    answers = []
    hop_values = []
    for _ in range(batch_size):
        cycle = torch.randperm(node_count, generator=generator)
        mapping = torch.empty(node_count, dtype=torch.long)
        mapping[cycle] = cycle.roll(-1)
        start = int(torch.randint(node_count, (1,), generator=generator))
        hops = (
            fixed_hops
            if fixed_hops is not None
            else int(
                torch.randint(
                    min_hops,
                    max_hops + 1,
                    (1,),
                    generator=generator,
                )
            )
        )
        answer = start
        for _ in range(hops):
            answer = int(mapping[answer])
        row = [1]
        for node, target in enumerate(mapping.tolist()):
            row.extend((node_offset + node, node_offset + target))
        row.extend(
            (
                2,
                hop_offset + hops - min_hops,
                node_offset + start,
                node_offset + answer,
                3,
            )
        )
        rows.append(row)
        answers.append(node_offset + answer)
        hop_values.append(hops)

    input_ids = torch.tensor(rows, dtype=torch.long, device=device)
    labels = torch.full_like(input_ids, -100)
    answer_position = 2 * node_count + 4
    labels[:, answer_position] = input_ids[:, answer_position]
    prediction_positions = torch.full(
        (batch_size,), answer_position - 1, dtype=torch.long, device=device
    )
    return PointerChaseBatch(
        input_ids=input_ids,
        labels=labels,
        prediction_positions=prediction_positions,
        answers=torch.tensor(answers, dtype=torch.long, device=device),
        hops=torch.tensor(hop_values, dtype=torch.long, device=device),
    )
