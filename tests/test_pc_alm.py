import torch

from minillm.aira import (
    backprop_gradients,
    gradient_cosine,
    local_augmented_gradients,
    minimum_layer_cosine,
)


def weights(depth: int, width: int, seed: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    return tuple(
        (torch.randn(width, width) / width**0.5).requires_grad_() for _ in range(depth)
    )


def test_pc_alm_improves_finite_inference_alignment() -> None:
    model = weights(8, 8, 50)
    inputs = torch.randn(1, 8)
    targets = torch.randn(1, 8)
    bp = backprop_gradients(model, inputs, targets, activation="tanh")
    pc = local_augmented_gradients(
        model,
        inputs,
        targets,
        inference_steps=16,
        state_step_size=0.2,
        dual_rate=0.0,
        activation="tanh",
    )
    alm = local_augmented_gradients(
        model,
        inputs,
        targets,
        inference_steps=16,
        state_step_size=0.2,
        dual_rate=0.5,
        activation="tanh",
    )
    assert gradient_cosine(alm.gradients, bp) > gradient_cosine(pc.gradients, bp)
    assert minimum_layer_cosine(alm.gradients, bp) > 0.5
    assert alm.residual_norm < 0.2


def test_pc_alm_gradients_are_local_shape_and_finite() -> None:
    model = weights(4, 6, 7)
    result = local_augmented_gradients(
        model,
        torch.randn(2, 6),
        torch.randn(2, 6),
        inference_steps=8,
        state_step_size=0.1,
        dual_rate=0.2,
        activation="linear",
    )
    assert len(result.gradients) == len(model)
    for gradient, weight in zip(result.gradients, model):
        assert gradient.shape == weight.shape
        assert torch.isfinite(gradient).all()
