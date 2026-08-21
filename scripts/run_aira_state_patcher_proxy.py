#!/usr/bin/env python3
"""Train a bounded synthetic proxy for AIra event-state catch-up."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from minillm.aira.state_patcher import RecurrentStatePatcher, state_patch_loss


class SyntheticDynamics:
    def __init__(
        self,
        *,
        layers: int,
        state_dim: int,
        event_dim: int,
        byte_dim: int,
        future_vocab: int,
        seed: int,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.event_weight = torch.randn(event_dim, state_dim, generator=generator) / event_dim**0.5
        self.byte_table = torch.randn(256, byte_dim, generator=generator) / byte_dim**0.5
        self.byte_weight = torch.randn(byte_dim, state_dim, generator=generator) / byte_dim**0.5
        self.layer_bias = torch.randn(layers, state_dim, generator=generator) * 0.2
        self.future_weight = torch.randn(state_dim, future_vocab, generator=generator) / state_dim**0.5

    def target(
        self,
        state: torch.Tensor,
        event: torch.Tensor,
        emitted: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        byte_features = self.byte_table[emitted].mean(dim=1)
        update = torch.tanh(
            event @ self.event_weight
            + byte_features @ self.byte_weight
        ).unsqueeze(1)
        update = 0.45 * torch.tanh(update + self.layer_bias.unsqueeze(0))
        return torch.where(mask.unsqueeze(-1), state + update, state)

    def future_logits(self, state: torch.Tensor) -> torch.Tensor:
        return state.mean(dim=1) @ self.future_weight


def sample_batch(
    generator: torch.Generator,
    *,
    batch: int,
    layers: int,
    state_dim: int,
    event_dim: int,
    span: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    state = torch.randn(batch, layers, state_dim, generator=generator)
    event = torch.randn(batch, event_dim, generator=generator)
    emitted = torch.randint(0, 256, (batch, span), generator=generator)
    mask = torch.rand(batch, layers, generator=generator) > 0.25
    mask[:, 0] = False  # an attention-like anchor remains exact
    mask[:, -1] = True
    return state, event, emitted, mask


def evaluate(
    model: RecurrentStatePatcher,
    dynamics: SyntheticDynamics,
    batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, float]:
    state, event, emitted, mask = batch_data
    target = dynamics.target(state, event, emitted, mask)
    with torch.inference_mode():
        output = model(state, event, emitted, patch_mask=mask)
        mse = (output.state[mask] - target[mask]).square().mean()
        baseline = (state[mask] - target[mask]).square().mean()
        future = torch.nn.functional.kl_div(
            torch.log_softmax(dynamics.future_logits(output.state), dim=-1),
            torch.softmax(dynamics.future_logits(target), dim=-1),
            reduction="batchmean",
        )
        anchor_error = (output.state[~mask] - state[~mask]).abs().max()
        return {
            "state_mse": float(mse),
            "zero_delta_baseline_mse": float(baseline),
            "mse_ratio": float(mse / baseline),
            "future_kl": float(future),
            "anchor_max_abs_error": float(anchor_error),
            "mean_confidence": float(output.confidence[mask].mean()),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200, choices=range(1, 301))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--output", default="results/aira_state_patcher_proxy.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    layers, state_dim, event_dim, byte_dim = 6, 24, 16, 12
    model = RecurrentStatePatcher(
        layers=layers,
        state_dim=state_dim,
        event_dim=event_dim,
        hidden_dim=64,
        byte_dim=byte_dim,
    )
    dynamics = SyntheticDynamics(
        layers=layers,
        state_dim=state_dim,
        event_dim=event_dim,
        byte_dim=byte_dim,
        future_vocab=48,
        seed=args.seed + 1,
    )
    train_generator = torch.Generator().manual_seed(args.seed + 2)
    eval_generator = torch.Generator().manual_seed(args.seed + 3)
    evaluation_batch = sample_batch(
        eval_generator,
        batch=256,
        layers=layers,
        state_dim=state_dim,
        event_dim=event_dim,
        span=8,
    )
    initial = evaluate(model, dynamics, evaluation_batch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    samples = []
    started = time.perf_counter()
    model.train()
    for step in range(args.steps):
        state, event, emitted, mask = sample_batch(
            train_generator,
            batch=args.batch_size,
            layers=layers,
            state_dim=state_dim,
            event_dim=event_dim,
            span=8,
        )
        target = dynamics.target(state, event, emitted, mask)
        output = model(state, event, emitted, patch_mask=mask)
        loss = state_patch_loss(
            output,
            target,
            mask,
            student_future_logits=dynamics.future_logits(output.state),
            teacher_future_logits=dynamics.future_logits(target),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in {0, args.steps // 4, args.steps // 2, 3 * args.steps // 4, args.steps - 1}:
            samples.append(
                {
                    "step": step + 1,
                    "total_loss": float(loss.total.detach()),
                    "state_mse": float(loss.state_mse.detach()),
                    "future_kl": float(loss.future_kl.detach()),
                    "gradient_norm": float(gradient_norm),
                }
            )
    elapsed = time.perf_counter() - started
    final = evaluate(model, dynamics, evaluation_batch)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    report = {
        "schema_version": 1,
        "experiment": "synthetic event-state catch-up proxy",
        "scope_warning": (
            "This validates the patcher training path on known synthetic dynamics; "
            "it is not evidence that Qwen recurrent states can yet be reconstructed."
        ),
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "threads": args.threads,
        "training_seconds": elapsed,
        "parameter_count": parameter_count,
        "layers": layers,
        "state_dim": state_dim,
        "event_dim": event_dim,
        "initial": initial,
        "final": final,
        "samples": samples,
        "success_gate": {
            "mse_ratio_below_0_25": final["mse_ratio"] < 0.25,
            "anchors_exact": final["anchor_max_abs_error"] == 0.0,
            "future_kl_below_initial": final["future_kl"] < initial["future_kl"],
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
