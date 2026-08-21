from __future__ import annotations

import torch

from minillm.aira.event_core import MultiByteEventLM, multi_byte_event_loss


def test_multi_byte_event_core_shapes_and_masked_loss() -> None:
    torch.manual_seed(12)
    model = MultiByteEventLM(
        vocab_size=64,
        context_size=6,
        d_model=12,
        maximum_bytes=4,
        route_actions=5,
    )
    contexts = torch.randint(0, 64, (3, 6))
    byte_targets = torch.randint(0, 256, (3, 4))
    lengths = torch.tensor([1, 3, 4])
    routes = torch.tensor([0, 1, 2])

    output = model(contexts)
    losses = multi_byte_event_loss(output, byte_targets, lengths, routes)
    losses.total.backward()

    assert output.byte_logits.shape == (3, 4, 256)
    assert output.continuation_logits.shape == (3, 3)
    assert output.route_logits.shape == (3, 5)
    assert torch.isfinite(losses.total)
    assert model.embedding.weight.grad is not None


def test_copy_routes_can_disable_literal_and_stop_supervision() -> None:
    model = MultiByteEventLM(vocab_size=32, context_size=4, d_model=8, maximum_bytes=3)
    output = model(torch.randint(0, 32, (2, 4)))
    losses = multi_byte_event_loss(
        output,
        torch.zeros(2, 3, dtype=torch.long),
        torch.tensor([3, 3]),
        torch.tensor([1, 2]),
        byte_supervised=torch.zeros(2, dtype=torch.bool),
    )

    assert losses.byte.item() == 0
    assert losses.continuation.item() == 0
    assert torch.isfinite(losses.total)


def test_single_byte_head_has_zero_continuation_loss() -> None:
    model = MultiByteEventLM(vocab_size=32, context_size=4, d_model=8, maximum_bytes=1)
    output = model(torch.randint(0, 32, (2, 4)))
    losses = multi_byte_event_loss(
        output,
        torch.randint(0, 256, (2, 1)),
        torch.ones(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
    )

    assert output.continuation_logits.shape == (2, 0)
    assert losses.continuation.item() == 0
