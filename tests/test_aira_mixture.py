from __future__ import annotations

import pytest
import torch

from minillm.aira import normalized_shelf_neural_mixture, shelf_distribution


def test_shelf_distribution_is_normalized_and_preserves_top_mass() -> None:
    top_tokens = torch.tensor([1, 3])
    confidence = torch.tensor([0.7, 0.4])
    probabilities = shelf_distribution(top_tokens, confidence, vocab_size=4)

    assert probabilities.sum(dim=-1).tolist() == pytest.approx([1.0, 1.0])
    assert probabilities[0].tolist() == pytest.approx([0.1, 0.7, 0.1, 0.1])
    assert probabilities[1].tolist() == pytest.approx([0.2, 0.2, 0.2, 0.4])


def test_normalized_mixture_has_exact_endpoints() -> None:
    shelf = shelf_distribution(torch.tensor([2]), torch.tensor([0.8]), 4)
    logits = torch.tensor([[1.0, 2.0, -1.0, 0.0]])

    shelf_only = normalized_shelf_neural_mixture(shelf, logits, 1.0)
    neural_only = normalized_shelf_neural_mixture(shelf, logits, 0.0)
    middle = normalized_shelf_neural_mixture(shelf, logits, 0.25)

    assert torch.allclose(shelf_only, shelf)
    assert torch.allclose(neural_only, torch.softmax(logits, dim=-1))
    assert torch.allclose(middle.sum(dim=-1), torch.ones(1))


def test_soft_mixture_retains_neural_gradient() -> None:
    shelf = shelf_distribution(torch.tensor([0]), torch.tensor([0.9]), 3)
    logits = torch.zeros(1, 3, requires_grad=True)
    mixture = normalized_shelf_neural_mixture(shelf, logits, 0.75)

    (-torch.log(mixture[0, 1])).backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.linalg.vector_norm(logits.grad) > 0


def test_invalid_shelf_probability_mass_is_rejected() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        normalized_shelf_neural_mixture(
            torch.tensor([[0.2, 0.2]]), torch.zeros(1, 2), 0.5
        )
