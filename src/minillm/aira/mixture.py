"""Normalized distributions for shelf predictions and shelf/neural mixtures."""

from __future__ import annotations

import torch


def shelf_distribution(
    top_tokens: torch.Tensor,
    top_probabilities: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """Expand a shelf's top-token confidence into a normalized distribution.

    A compact shelf stores only the most frequent continuation and its empirical
    probability. The unresolved mass is distributed uniformly over non-top
    symbols. This conservative maximum-entropy tail makes shelf-only bypasses
    scoreable with proper log loss without pretending that the tail was stored.
    """

    if vocab_size < 2:
        raise ValueError("vocab_size must be at least two")
    if top_tokens.shape != top_probabilities.shape:
        raise ValueError("top_tokens and top_probabilities must have the same shape")
    if not top_probabilities.is_floating_point():
        raise TypeError("top_probabilities must be floating point")
    if torch.any((top_tokens < 0) | (top_tokens >= vocab_size)):
        raise ValueError("top token is outside the vocabulary")
    if torch.any((top_probabilities < 0) | (top_probabilities > 1)):
        raise ValueError("top probabilities must lie in [0, 1]")

    tail = (1.0 - top_probabilities) / (vocab_size - 1)
    probabilities = tail.unsqueeze(-1).expand(*tail.shape, vocab_size).clone()
    indices = top_tokens.to(device=top_probabilities.device, dtype=torch.long)
    probabilities.scatter_(-1, indices.unsqueeze(-1), top_probabilities.unsqueeze(-1))
    return probabilities


def normalized_shelf_neural_mixture(
    shelf_probabilities: torch.Tensor,
    neural_logits: torch.Tensor,
    shelf_weight: torch.Tensor | float,
) -> torch.Tensor:
    """Return a convex, normalized shelf/neural probability mixture.

    ``shelf_weight=1`` is a shelf-only decision; ``0`` is neural-only. Intermediate
    values support calibrated soft deferral. Neural logits are converted with a
    softmax before mixing, so this function never mixes incomparable confidence
    scores or reports a hybrid pseudo-perplexity.
    """

    if shelf_probabilities.shape != neural_logits.shape:
        raise ValueError("shelf probabilities and neural logits must have equal shape")
    shelf_probabilities = shelf_probabilities.to(
        device=neural_logits.device, dtype=neural_logits.dtype
    )
    if torch.any(shelf_probabilities < 0):
        raise ValueError("shelf probabilities cannot be negative")
    shelf_mass = shelf_probabilities.sum(dim=-1, keepdim=True)
    if not torch.allclose(
        shelf_mass, torch.ones_like(shelf_mass), atol=1e-5, rtol=1e-5
    ):
        raise ValueError("shelf probabilities must sum to one")

    weight = torch.as_tensor(
        shelf_weight,
        dtype=neural_logits.dtype,
        device=neural_logits.device,
    )
    if torch.any((weight < 0) | (weight > 1)):
        raise ValueError("shelf weight must lie in [0, 1]")
    if weight.ndim == neural_logits.ndim - 1:
        weight = weight.unsqueeze(-1)
    neural_probabilities = torch.softmax(neural_logits, dim=-1)
    return weight * shelf_probabilities + (1.0 - weight) * neural_probabilities
