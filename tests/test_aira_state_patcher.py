from __future__ import annotations

import pytest
import torch

from minillm.aira.state_patcher import RecurrentStatePatcher, state_patch_loss


def test_state_patcher_preserves_anchor_layers() -> None:
    torch.manual_seed(3)
    model = RecurrentStatePatcher(
        layers=4, state_dim=8, event_dim=6, hidden_dim=16, byte_dim=5
    )
    state = torch.randn(2, 4, 8)
    event = torch.randn(2, 6)
    emitted = torch.randint(0, 256, (2, 5))
    mask = torch.tensor([[False, True, True, False], [False, True, False, True]])

    output = model(state, event, emitted, patch_mask=mask)

    assert output.state.shape == state.shape
    assert output.confidence.shape == mask.shape
    assert torch.equal(output.state[~mask], state[~mask])
    assert torch.equal(output.delta[~mask], torch.zeros_like(output.delta[~mask]))
    assert torch.equal(output.confidence[~mask], torch.ones_like(output.confidence[~mask]))


def test_state_patch_loss_supervises_state_future_and_confidence() -> None:
    torch.manual_seed(4)
    model = RecurrentStatePatcher(
        layers=3, state_dim=7, event_dim=5, hidden_dim=14, byte_dim=4
    )
    state = torch.randn(3, 3, 7)
    event = torch.randn(3, 5)
    emitted = torch.randint(0, 256, (3, 4))
    mask = torch.tensor(
        [[False, True, True], [False, True, True], [False, True, False]]
    )
    output = model(state, event, emitted, patch_mask=mask)
    target = output.state.detach() + mask.unsqueeze(-1) * 0.1
    future_projection = torch.randn(7, 11)

    loss = state_patch_loss(
        output,
        target,
        mask,
        student_future_logits=output.state.mean(dim=1) @ future_projection,
        teacher_future_logits=target.mean(dim=1) @ future_projection,
    )

    assert torch.isfinite(loss.total)
    assert loss.total > 0
    loss.total.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_state_patcher_rejects_empty_patch_mask() -> None:
    model = RecurrentStatePatcher(
        layers=2, state_dim=4, event_dim=3, hidden_dim=8
    )
    state = torch.randn(1, 2, 4)
    output = model(
        state,
        torch.randn(1, 3),
        torch.randint(0, 256, (1, 2)),
        patch_mask=torch.zeros(1, 2, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="at least one"):
        state_patch_loss(output, state, torch.zeros(1, 2, dtype=torch.bool))
