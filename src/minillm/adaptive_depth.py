"""Losses and diagnostics for one-weight, variable-compute recurrent models."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_depth_consistency_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Distill a deeper recurrence into a shallower pass at supervised positions.

    The teacher is detached by construction. ``labels`` follows the decoder convention:
    token ``t`` is predicted by logits at ``t - 1`` and ``-100`` marks ignored targets.
    """

    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have the same shape")
    if labels.shape != student_logits.shape[:2]:
        raise ValueError("labels must match the batch and time dimensions")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if student_logits.shape[1] < 2:
        raise ValueError("depth consistency requires at least two tokens")

    valid = labels[:, 1:] != -100
    if not valid.any():
        return student_logits.sum() * 0
    student = student_logits[:, :-1][valid].float() / temperature
    teacher = teacher_logits[:, :-1][valid].detach().float() / temperature
    teacher_probabilities = F.softmax(teacher, dim=-1)
    teacher_log_probabilities = F.log_softmax(teacher, dim=-1)
    student_log_probabilities = F.log_softmax(student, dim=-1)
    kl = torch.sum(
        teacher_probabilities * (teacher_log_probabilities - student_log_probabilities),
        dim=-1,
    )
    return kl.mean() * temperature**2
