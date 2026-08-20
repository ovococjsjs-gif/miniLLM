#!/usr/bin/env python3
"""Single-GPU entry point for a pinned 20M L1 training checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

from minillm.analysis import profile_model
from minillm.config import MiniLLMConfig
from minillm.generation import load_model_checkpoint, save_inference_checkpoint
from minillm.training import TrainConfig, train_proxy


def git_provenance() -> tuple[str, bool]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"], text=True
    )
    return commit, bool(status)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["configs/l1_attention_20m.json", "configs/l1_edge_20m.json"],
        default="configs/l1_attention_20m.json",
    )
    parser.add_argument("--tokens", default="data/tokens-github-pilot")
    parser.add_argument("--data-report", default="results/github_pilot_data.json")
    parser.add_argument(
        "--tokenizer-manifest",
        default="artifacts/tokenizer-github-pilot-v1/manifest.json",
    )
    parser.add_argument("--output", default="runs/l1-20m")
    parser.add_argument("--target-tokens", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.01)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--fused-optimizer", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--resume")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="allow a non-clean worktree while recording that fact in provenance",
    )
    args = parser.parse_args()

    commit, git_dirty = git_provenance()
    if git_dirty and not (args.dry_run or args.allow_dirty):
        raise RuntimeError(
            "L1 training requires a clean git worktree; commit changes or use --allow-dirty"
        )
    model_config = MiniLLMConfig.load(args.model)
    data_report = json.loads(Path(args.data_report).read_text(encoding="utf-8"))
    tokenizer_manifest = json.loads(
        Path(args.tokenizer_manifest).read_text(encoding="utf-8")
    )
    train_tokens_available = data_report["packed_tokens"]["splits"]["train"]["tokens"]
    target_tokens = args.target_tokens or train_tokens_available
    tokens_per_step = (
        args.batch_size * args.sequence_length * args.gradient_accumulation
    )
    steps = math.ceil(target_tokens / tokens_per_step)
    warmup_tokens = max(tokens_per_step, int(target_tokens * args.warmup_ratio))
    training = TrainConfig(
        steps=steps,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_steps=0,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed,
        device=args.device,
        precision=args.precision,
        gradient_checkpointing=args.gradient_checkpointing,
        fused_optimizer=args.fused_optimizer,
        schedule_tokens=target_tokens,
        warmup_tokens=warmup_tokens,
    )
    metadata = {
        "stage": "l1-20m",
        "git_commit": commit,
        "git_dirty": git_dirty,
        "model_config": args.model,
        "corpus_sha256": data_report["corpus"]["corpus_sha256"],
        "train_token_sha256": data_report["packed_tokens"]["splits"]["train"]["sha256"],
        "validation_token_sha256": data_report["packed_tokens"]["splits"]["validation"][
            "sha256"
        ],
        "tokenizer_sha256": tokenizer_manifest["tokenizer_sha256"],
        "target_tokens": target_tokens,
    }
    plan = {
        "model": args.model,
        "parameters": profile_model(model_config).unique_parameters,
        "device": args.device,
        "precision": args.precision,
        "gradient_checkpointing": args.gradient_checkpointing,
        "fused_optimizer": args.fused_optimizer,
        "tokens_per_step": tokens_per_step,
        "steps": steps,
        "target_tokens": target_tokens,
        "actual_planned_tokens": steps * tokens_per_step,
        "warmup_tokens": warmup_tokens,
        "metadata": metadata,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    output = Path(args.output)
    if output.exists() and args.resume is None:
        raise FileExistsError("refusing to overwrite an existing L1 run")
    summary = train_proxy(
        model_config,
        training,
        train_tokens=Path(args.tokens) / "train.bin",
        validation_tokens=Path(args.tokens) / "validation.bin",
        output_directory=output,
        resume_from=args.resume,
        run_metadata=metadata,
    )
    best = load_model_checkpoint(output / "best.pt", device="cpu")
    inference_path = output / "best-inference.pt"
    if not inference_path.exists():
        save_inference_checkpoint(
            inference_path,
            best.model,
            step=best.step,
            metadata=metadata,
        )
    payload = {
        "plan": plan,
        "summary": summary,
        "inference_checkpoint": str(inference_path),
    }
    (output / "l1-summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
