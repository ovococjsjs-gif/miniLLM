import torch

from minillm.distillation import (
    CandidateTrajectory,
    decoupled_topk_kl,
    extract_teacher_topk,
    select_capacity_aligned_trajectory,
    stepwise_distillation_weights,
)


def test_decoupled_topk_loss_is_zero_for_identical_distributions() -> None:
    torch.manual_seed(12)
    logits = torch.randn(2, 3, 20)
    teacher = extract_teacher_topk(logits, k=5)
    loss = decoupled_topk_kl(logits.clone().requires_grad_(), teacher, temperature=1.0)
    assert abs(float(loss.detach())) < 1e-5
    loss.backward()


def test_decoupled_topk_preserves_tail_mass_signal() -> None:
    teacher_logits = torch.tensor([[5.0, 4.0, 1.0, 0.0]])
    teacher = extract_teacher_topk(teacher_logits, k=2)
    matched = decoupled_topk_kl(teacher_logits, teacher, temperature=1.0)
    wrong_tail = decoupled_topk_kl(
        torch.tensor([[5.0, 4.0, 8.0, 8.0]]), teacher, temperature=1.0
    )
    assert wrong_tail > matched + 0.1


def test_capacity_selection_and_step_weights() -> None:
    selected = select_capacity_aligned_trajectory(
        [
            CandidateTrajectory("wrong", False, 0.1, 10),
            CandidateTrajectory("too-hard", True, 8.0, 100),
            CandidateTrajectory("learnable", True, 2.0, 20),
        ]
    )
    assert selected.identifier == "learnable"
    weights = stepwise_distillation_weights(torch.tensor([0.0, 0.5, 4.0]))
    assert weights[0] == 1
    assert weights[0] > weights[1] > weights[2]
    assert weights[2] >= 0.05
