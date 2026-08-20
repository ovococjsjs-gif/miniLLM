"""Capacity-aware sparse-logit and trajectory distillation primitives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class TeacherTopK:
    indices: torch.Tensor
    logits: torch.Tensor
    mass: torch.Tensor


def extract_teacher_topk(teacher_logits: torch.Tensor, k: int = 32) -> TeacherTopK:
    if not 1 <= k < teacher_logits.shape[-1]:
        raise ValueError("k must be between 1 and vocabulary size - 1")
    logits, indices = teacher_logits.topk(k, dim=-1)
    log_probabilities = F.log_softmax(teacher_logits.float(), dim=-1)
    selected = log_probabilities.gather(-1, indices)
    mass = selected.logsumexp(dim=-1).exp()
    return TeacherTopK(indices, logits, mass)


def decoupled_topk_kl(
    student_logits: torch.Tensor,
    teacher: TeacherTopK,
    *,
    temperature: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """KL decomposition into Top-K membership and conditional Top-K shape.

    Temperature affects only the conditional distribution within the teacher's
    retained support. The untempered binary term preserves the missing tail mass.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if teacher.indices.shape != teacher.logits.shape:
        raise ValueError("teacher indices/logits shapes differ")
    if student_logits.shape[:-1] != teacher.indices.shape[:-1]:
        raise ValueError("student and teacher prefix shapes differ")
    if teacher.mass.shape != teacher.indices.shape[:-1]:
        raise ValueError("teacher mass shape is invalid")

    eps = torch.finfo(torch.float32).eps
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    selected_student_log_probs = student_log_probs.gather(-1, teacher.indices)
    student_mass = (
        selected_student_log_probs.logsumexp(dim=-1).exp().clamp(eps, 1 - eps)
    )
    teacher_mass = teacher.mass.float().clamp(eps, 1 - eps)
    binary_kl = teacher_mass * (teacher_mass.log() - student_mass.log()) + (
        1 - teacher_mass
    ) * ((1 - teacher_mass).log() - (1 - student_mass).log())

    teacher_conditional = F.softmax(teacher.logits.float() / temperature, dim=-1)
    teacher_conditional_log = F.log_softmax(
        teacher.logits.float() / temperature, dim=-1
    )
    selected_student_logits = student_logits.float().gather(-1, teacher.indices)
    student_conditional_log = F.log_softmax(
        selected_student_logits / temperature, dim=-1
    )
    conditional_kl = torch.sum(
        teacher_conditional * (teacher_conditional_log - student_conditional_log),
        dim=-1,
    )
    loss = binary_kl + teacher_mass * temperature**2 * conditional_kl
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError("reduction must be none, sum, or mean")


def mixed_lm_distillation_loss(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    teacher: TeacherTopK,
    *,
    distillation_weight: float = 0.5,
    temperature: float = 2.0,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 0 <= distillation_weight <= 1:
        raise ValueError("distillation_weight must be in [0, 1]")
    language_model_loss = F.cross_entropy(
        student_logits.reshape(-1, student_logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )
    token_losses = decoupled_topk_kl(
        student_logits, teacher, temperature=temperature, reduction="none"
    )
    valid = labels != ignore_index
    distillation_loss = (
        token_losses[valid].mean() if valid.any() else token_losses.mean() * 0
    )
    total = (
        1 - distillation_weight
    ) * language_model_loss + distillation_weight * distillation_loss
    return total, language_model_loss, distillation_loss


@dataclass(frozen=True)
class CandidateTrajectory:
    identifier: str
    correct: bool
    student_nll: float
    reasoning_tokens: int
    tool_errors: int = 0


def select_capacity_aligned_trajectory(
    candidates: list[CandidateTrajectory],
    *,
    token_penalty: float = 0.001,
    tool_error_penalty: float = 5.0,
) -> CandidateTrajectory:
    """Select the easiest correct teacher path under the student's own distribution."""

    correct = [candidate for candidate in candidates if candidate.correct]
    if not correct:
        raise ValueError("no correct teacher trajectory")
    return min(
        correct,
        key=lambda item: (
            item.student_nll
            + token_penalty * item.reasoning_tokens
            + tool_error_penalty * item.tool_errors,
            item.reasoning_tokens,
            item.identifier,
        ),
    )


def stepwise_distillation_weights(
    divergence: torch.Tensor,
    *,
    scale: float = 2.0,
    minimum: float = 0.05,
) -> torch.Tensor:
    """Attenuate teacher supervision after student/tool state divergence."""

    if scale < 0 or not 0 <= minimum <= 1:
        raise ValueError("invalid weighting parameters")
    return torch.exp(-scale * divergence.float().clamp_min(0)).clamp_min(minimum)
