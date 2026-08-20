from __future__ import annotations

import pytest
import torch

from minillm.aira import (
    AttentionByteEventLM,
    ByteEventLM,
    ConvByteEventLM,
    EventContextLM,
)


def test_event_context_lm_is_bounded_and_differentiable() -> None:
    torch.manual_seed(4)
    model = EventContextLM(vocab_size=32, context_size=6, d_model=12)
    contexts = torch.randint(0, 32, (5, 6))
    targets = torch.randint(0, 32, (5,))

    logits = model(contexts)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()

    assert logits.shape == (5, 32)
    assert model.embedding.weight.grad is not None
    assert model.parameter_bytes > 0


def test_byte_event_lm_predicts_raw_byte_distribution() -> None:
    model = ByteEventLM(vocab_size=64, context_size=5, d_model=8)
    logits = model(torch.randint(0, 64, (3, 5)))

    assert logits.shape == (3, 256)
    assert model.parameter_bytes > 0


@pytest.mark.parametrize(
    "model_class", [ByteEventLM, ConvByteEventLM, AttentionByteEventLM]
)
def test_all_byte_event_cores_have_matched_interface(model_class) -> None:
    model = model_class(vocab_size=64, context_size=8, d_model=16)
    logits = model(torch.randint(0, 64, (3, 8)))

    assert logits.shape == (3, 256)
    assert model.parameter_bytes > 0


def test_event_context_lm_rejects_wrong_context_shape() -> None:
    model = EventContextLM(vocab_size=16, context_size=4, d_model=8)
    with pytest.raises(ValueError, match="context_ids"):
        model(torch.ones(2, 3, dtype=torch.long))
