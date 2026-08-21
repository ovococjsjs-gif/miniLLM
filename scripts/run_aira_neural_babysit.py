#!/usr/bin/env python3
"""Train and evaluate a real parameterized AI Babysit output adapter.

Unlike the deterministic SkillShelf, this experiment disables stored-answer routes.
Teacher corrections supervise a small neural residual on frozen Qwen hidden states;
the residual changes logits during autoregressive generation. A learned prompt gate
is trained jointly to preserve unrelated requests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Collected:
    hidden: torch.Tensor
    targets: torch.Tensor
    base_logits: torch.Tensor
    other_logsumexp: torch.Tensor
    other_max: torch.Tensor
    candidates: np.ndarray
    target_indices: torch.Tensor
    index: list[dict[str, str]]
    metadata: dict[str, int]


class OutputResidual(nn.Module):
    def __init__(self, embedding: int, hidden: int, candidates: int) -> None:
        super().__init__()
        self.input = nn.Linear(embedding, hidden)
        self.output = nn.Linear(hidden, candidates)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.output(F.silu(self.input(hidden)))


class PromptGate(nn.Module):
    def __init__(self, embedding: int, hidden: int = 32) -> None:
        super().__init__()
        self.input = nn.Linear(embedding, hidden)
        self.output = nn.Linear(hidden, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.output(torch.tanh(self.input(hidden))).squeeze(-1)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_tsv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for identity, prompt, answer in rows:
            handle.write(
                f"{identity}\t{prompt.encode().hex()}\t{answer.encode().hex()}\n"
            )


def read_summary(path: Path) -> dict[str, int]:
    return {
        key: int(value)
        for key, value in (
            line.split("\t") for line in path.read_text(encoding="utf-8").splitlines()
        )
    }


def load_collected(root: Path) -> Collected:
    metadata = read_summary(root / "summary.tsv")
    tokens = metadata["tokens"]
    embedding = metadata["embedding"]
    candidates_count = metadata["candidates"]
    candidates = np.fromfile(root / "candidates.i32.bin", dtype="<i4")
    if candidates.shape != (candidates_count,):
        raise ValueError("candidate token file shape mismatch")
    hidden = torch.from_numpy(
        np.fromfile(root / "hidden.f32.bin", dtype="<f4")
        .reshape(tokens, embedding)
        .copy()
    )
    targets_array = np.fromfile(root / "targets.i32.bin", dtype="<i4")
    targets = torch.from_numpy(targets_array.astype(np.int64))
    base_logits = torch.from_numpy(
        np.fromfile(root / "base_candidate_logits.f32.bin", dtype="<f4")
        .reshape(tokens, candidates_count)
        .copy()
    )
    other_logsumexp = torch.from_numpy(
        np.fromfile(root / "other_logsumexp.f32.bin", dtype="<f4").copy()
    )
    other_max = torch.from_numpy(
        np.fromfile(root / "other_max.f32.bin", dtype="<f4").copy()
    )
    if any(
        tensor.shape[0] != tokens
        for tensor in (targets, base_logits, other_logsumexp, other_max)
    ):
        raise ValueError("collected array length mismatch")
    lookup = {int(token): index for index, token in enumerate(candidates)}
    try:
        target_indices = torch.tensor([lookup[int(token)] for token in targets_array])
    except KeyError as error:
        raise ValueError("target token missing from candidate vocabulary") from error
    with (root / "index.tsv").open(encoding="utf-8") as handle:
        index = list(csv.DictReader(handle, delimiter="\t"))
    return Collected(
        hidden,
        targets,
        base_logits,
        other_logsumexp,
        other_max,
        candidates,
        target_indices,
        index,
        metadata,
    )


def prompt_states(collected: Collected) -> torch.Tensor:
    offsets = torch.tensor([int(row["offset"]) for row in collected.index])
    return collected.hidden[offsets]


def exact_metrics(
    model: OutputResidual,
    data: Collected,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> dict[str, float]:
    predictions = []
    with torch.no_grad():
        for start in range(0, data.metadata["tokens"], 128):
            hidden = (data.hidden[start : start + 128] - mean) * scale
            predictions.append(model(hidden))
    delta = torch.cat(predictions)
    adapted = data.base_logits + delta
    indices = torch.arange(data.metadata["tokens"])

    def scores(logits: torch.Tensor) -> tuple[float, float]:
        target = logits[indices, data.target_indices]
        denominator = torch.logaddexp(
            torch.logsumexp(logits, dim=-1), data.other_logsumexp
        )
        masked = logits.clone()
        masked[indices, data.target_indices] = -torch.inf
        competing = torch.maximum(masked.max(dim=-1).values, data.other_max)
        return float((denominator - target).mean()), float(
            (target > competing).float().mean()
        )

    base_nll, base_accuracy = scores(data.base_logits)
    adapted_nll, adapted_accuracy = scores(adapted)
    return {
        "tokens": float(data.metadata["tokens"]),
        "base_nll": base_nll,
        "adapted_nll": adapted_nll,
        "base_top1_accuracy": base_accuracy,
        "adapted_top1_accuracy": adapted_accuracy,
        "delta_rms": float(delta.square().mean().sqrt()),
        "delta_max_abs": float(delta.abs().max()),
    }


def gate_metrics(
    gate: PromptGate,
    states: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    expected_active: bool,
) -> dict[str, Any]:
    with torch.no_grad():
        probabilities = torch.sigmoid(gate((states - mean) * scale))
    decisions = probabilities >= 0.5
    expected = torch.full_like(decisions, expected_active)
    return {
        "records": int(states.shape[0]),
        "accuracy": float((decisions == expected).float().mean()),
        "active": int(decisions.sum()),
        "minimum_probability": float(probabilities.min()),
        "maximum_probability": float(probabilities.max()),
        "probabilities": [float(value) for value in probabilities],
    }


def save_adapter(
    path: Path,
    residual: OutputResidual,
    gate: PromptGate,
    candidates: np.ndarray,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    residual_state = residual.state_dict()
    gate_state = gate.state_dict()
    output_hidden, embedding = residual_state["input.weight"].shape
    candidate_count, _ = residual_state["output.weight"].shape
    gate_hidden, gate_embedding = gate_state["input.weight"].shape
    if gate_embedding != embedding or candidate_count != len(candidates):
        raise ValueError("adapter tensor dimensions disagree")
    with path.open("wb") as handle:
        handle.write(b"AIRAODA2")
        handle.write(
            struct.pack(
                "<IIIIff",
                embedding,
                output_hidden,
                candidate_count,
                gate_hidden,
                1.0,
                0.5,
            )
        )
        arrays = (
            np.asarray(candidates, dtype="<i4"),
            mean.detach().numpy().astype("<f4"),
            scale.detach().numpy().astype("<f4"),
            residual_state["input.weight"].detach().numpy().astype("<f4"),
            residual_state["input.bias"].detach().numpy().astype("<f4"),
            residual_state["output.weight"].detach().numpy().astype("<f4"),
            residual_state["output.bias"].detach().numpy().astype("<f4"),
            gate_state["input.weight"].detach().numpy().astype("<f4"),
            gate_state["input.bias"].detach().numpy().astype("<f4"),
            gate_state["output.weight"].detach().numpy().astype("<f4").reshape(-1),
            gate_state["output.bias"].detach().numpy().astype("<f4"),
        )
        for array in arrays:
            handle.write(array.tobytes())


def parse_generation(path: Path) -> dict[str, dict[str, Any]]:
    output = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            output[row["id"]] = {
                "answer": bytes.fromhex(row["answer_hex"]).decode(
                    "utf-8", errors="replace"
                ),
                "tokens": int(row["tokens"]),
                "gate_probability": float(row["gate_probability"]),
                "adapter_active": row["adapter_active"] == "1",
            }
    return output


def verify_answer(answer: str, task: dict[str, Any]) -> dict[str, Any]:
    lowered = answer.casefold()
    missing = [
        group
        for group in task["answer_requirements"]
        if not any(term.casefold() in lowered for term in group)
    ]
    forbidden = [
        term for term in task.get("forbidden", []) if term.casefold() in lowered
    ]
    return {
        "passed": bool(answer.strip()) and not missing and not forbidden,
        "missing_groups": missing,
        "forbidden_matches": forbidden,
    }


def run_command(command: list[str], log: Path) -> float:
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - started
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}"
        )
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/aira-one/neural_babysit_v1.json")
    parser.add_argument(
        "--controls", default="configs/aira-one/neural_adapter_controls_v1.json"
    )
    parser.add_argument(
        "--model",
        default="data/external/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M-github.gguf",
    )
    parser.add_argument(
        "--binary", default=".cache/qwen35-output-adapter/qwen35-output-adapter"
    )
    parser.add_argument("--cache", default=".cache/aira-neural-babysit-v1")
    parser.add_argument("--artifact", default="artifacts/aira-neural-babysit-v1")
    parser.add_argument("--output", default="results/aira_neural_babysit_v1.json")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    curriculum_path = Path(config["base_curriculum"])
    curriculum = json.loads(curriculum_path.read_text(encoding="utf-8"))
    controls_path = Path(args.controls)
    controls = json.loads(controls_path.read_text(encoding="utf-8"))
    tasks = {
        task["id"]: (cycle["id"], task)
        for cycle in curriculum["cycles"]
        for task in cycle["tasks"]
    }
    augmentations = config["train_augmentations"]
    if len(tasks) != 24 or set(augmentations) != set(tasks):
        raise ValueError("neural Babysit v1 requires 24 tasks and 24 augmentations")
    if config["steps"] > 300:
        raise ValueError("local neural Babysit training may not exceed 300 steps")

    cache = Path(args.cache)
    artifact = Path(args.artifact)
    cache.mkdir(parents=True, exist_ok=True)
    artifact.mkdir(parents=True, exist_ok=True)
    train_rows: list[tuple[str, str, str]] = []
    validation_rows: list[tuple[str, str, str]] = []
    task_by_validation_id: dict[str, dict[str, Any]] = {}
    for task_id, (cycle_id, task) in tasks.items():
        answer = task["answers"][task["language"]]
        train_rows.extend(
            (
                (f"{cycle_id}:{task_id}:base", task["train_prompt"], answer),
                (f"{cycle_id}:{task_id}:augment", augmentations[task_id], answer),
            )
        )
        validation_id = f"{cycle_id}:{task_id}"
        validation_rows.append((validation_id, task["validation_prompt"], answer))
        task_by_validation_id[validation_id] = task
    control_rows = [(item["id"], item["prompt"], "-") for item in controls["prompts"]]
    control_validation_rows = [
        row
        for row, item in zip(control_rows, controls["prompts"], strict=True)
        if item["split"] == "validation"
    ]
    write_tsv(artifact / "train.tsv", train_rows)
    write_tsv(artifact / "validation.tsv", validation_rows)
    write_tsv(artifact / "controls.tsv", control_rows)
    write_tsv(artifact / "controls_validation.tsv", control_validation_rows)

    binary = str(Path(args.binary))
    model = str(Path(args.model))
    train_cache = cache / "train"
    validation_cache = cache / "validation"
    controls_cache = cache / "controls"
    collection_seconds = {}
    collection_seconds["train"] = run_command(
        [binary, "collect", model, str(artifact / "train.tsv"), str(train_cache)],
        cache / "collect-train.log",
    )
    collection_seconds["validation"] = run_command(
        [
            binary,
            "collect",
            model,
            str(artifact / "validation.tsv"),
            str(validation_cache),
            str(train_cache / "candidates.i32.bin"),
        ],
        cache / "collect-validation.log",
    )
    collection_seconds["controls"] = run_command(
        [binary, "collect", model, str(artifact / "controls.tsv"), str(controls_cache)],
        cache / "collect-controls.log",
    )

    train = load_collected(train_cache)
    validation = load_collected(validation_cache)
    control_data = load_collected(controls_cache)
    if not np.array_equal(train.candidates, validation.candidates):
        raise ValueError("train and validation candidate vocabularies differ")
    torch.set_num_threads(2)
    torch.manual_seed(config["seed"])
    rng = random.Random(config["seed"])
    mean = train.hidden.mean(dim=0)
    standard_deviation = train.hidden.std(dim=0).clamp_min(0.05)
    scale = standard_deviation.reciprocal()
    residual = OutputResidual(
        train.metadata["embedding"], config["hidden_dim"], train.metadata["candidates"]
    )
    gate = PromptGate(train.metadata["embedding"])
    optimizer = torch.optim.AdamW(
        list(residual.parameters()) + list(gate.parameters()),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    positive_prompt_states = prompt_states(train)
    control_prompt_states = prompt_states(control_data)
    control_train_indices = torch.tensor(
        [
            index
            for index, item in enumerate(controls["prompts"])
            if item["split"] == "train"
        ]
    )
    control_validation_indices = torch.tensor(
        [
            index
            for index, item in enumerate(controls["prompts"])
            if item["split"] == "validation"
        ]
    )
    gate_train_states = torch.cat(
        (positive_prompt_states, control_prompt_states[control_train_indices])
    )
    gate_train_labels = torch.cat(
        (
            torch.ones(positive_prompt_states.shape[0]),
            torch.zeros(control_train_indices.shape[0]),
        )
    )

    training_samples = []
    started = time.perf_counter()
    residual.train()
    gate.train()
    for step in range(config["steps"]):
        batch = torch.tensor(
            [
                rng.randrange(train.metadata["tokens"])
                for _ in range(config["batch_size"])
            ]
        )
        hidden = (train.hidden[batch] - mean) * scale
        delta = residual(hidden)
        logits = train.base_logits[batch] + delta
        target = train.target_indices[batch]
        indices = torch.arange(batch.shape[0])
        target_logits = logits[indices, target]
        denominator = torch.logaddexp(
            torch.logsumexp(logits, dim=-1), train.other_logsumexp[batch]
        )
        token_nll = (denominator - target_logits).mean()
        gate_logits = gate((gate_train_states - mean) * scale)
        gate_bce = F.binary_cross_entropy_with_logits(gate_logits, gate_train_labels)
        delta_regularization = config["delta_regularization"] * delta.square().mean()
        loss = token_nll + 0.2 * gate_bce + delta_regularization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            list(residual.parameters()) + list(gate.parameters()), 1.0
        )
        optimizer.step()
        if step in {
            0,
            config["steps"] // 4,
            config["steps"] // 2,
            3 * config["steps"] // 4,
            config["steps"] - 1,
        }:
            training_samples.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "token_nll": float(token_nll.detach()),
                    "gate_bce": float(gate_bce.detach()),
                    "gradient_norm": float(gradient_norm),
                }
            )
    training_seconds = time.perf_counter() - started
    residual.eval()
    gate.eval()

    train_metrics = exact_metrics(residual, train, mean, scale)
    validation_metrics = exact_metrics(residual, validation, mean, scale)
    gate_results = {
        "positive_train": gate_metrics(gate, positive_prompt_states, mean, scale, True),
        "positive_validation": gate_metrics(
            gate, prompt_states(validation), mean, scale, True
        ),
        "negative_train": gate_metrics(
            gate,
            control_prompt_states[control_train_indices],
            mean,
            scale,
            False,
        ),
        "negative_validation": gate_metrics(
            gate,
            control_prompt_states[control_validation_indices],
            mean,
            scale,
            False,
        ),
    }

    checkpoint = artifact / "model.pt"
    torch.save(
        {
            "schema_version": 1,
            "residual": residual.state_dict(),
            "gate": gate.state_dict(),
            "mean": mean,
            "scale": scale,
            "candidates": torch.from_numpy(train.candidates.copy()),
            "config_sha256": sha256(config_path),
            "curriculum_sha256": sha256(curriculum_path),
        },
        checkpoint,
    )
    adapter_path = artifact / "adapter.bin"
    save_adapter(adapter_path, residual, gate, train.candidates, mean, scale)

    generation_seconds = {}
    base_validation_path = cache / "base-validation.tsv"
    adapted_validation_path = cache / "adapted-validation.tsv"
    base_controls_path = cache / "base-controls.tsv"
    adapted_controls_path = cache / "adapted-controls.tsv"
    generation_seconds["base_validation"] = run_command(
        [
            binary,
            "generate",
            model,
            str(artifact / "validation.tsv"),
            str(base_validation_path),
            "-",
            "128",
        ],
        cache / "generate-base-validation.log",
    )
    generation_seconds["adapted_validation"] = run_command(
        [
            binary,
            "generate",
            model,
            str(artifact / "validation.tsv"),
            str(adapted_validation_path),
            str(adapter_path),
            "128",
        ],
        cache / "generate-adapted-validation.log",
    )
    generation_seconds["base_controls"] = run_command(
        [
            binary,
            "generate",
            model,
            str(artifact / "controls_validation.tsv"),
            str(base_controls_path),
            "-",
            "64",
        ],
        cache / "generate-base-controls.log",
    )
    generation_seconds["adapted_controls"] = run_command(
        [
            binary,
            "generate",
            model,
            str(artifact / "controls_validation.tsv"),
            str(adapted_controls_path),
            str(adapter_path),
            "64",
        ],
        cache / "generate-adapted-controls.log",
    )

    base_validation = parse_generation(base_validation_path)
    adapted_validation = parse_generation(adapted_validation_path)
    validation_records = []
    for identity, _, _ in validation_rows:
        task = task_by_validation_id[identity]
        base = base_validation[identity]
        adapted = adapted_validation[identity]
        base_check = verify_answer(base["answer"], task)
        adapted_check = verify_answer(adapted["answer"], task)
        validation_records.append(
            {
                "id": identity,
                "prompt": task["validation_prompt"],
                "base": {**base, "verification": base_check},
                "adapted": {**adapted, "verification": adapted_check},
            }
        )
    base_controls = parse_generation(base_controls_path)
    adapted_controls = parse_generation(adapted_controls_path)
    control_records = []
    for identity, prompt, _ in control_validation_rows:
        base = base_controls[identity]
        adapted = adapted_controls[identity]
        control_records.append(
            {
                "id": identity,
                "prompt": prompt,
                "base": base,
                "adapted": adapted,
                "byte_exact_preserved": base["answer"] == adapted["answer"],
            }
        )

    parameter_count = sum(
        parameter.numel()
        for parameter in list(residual.parameters()) + list(gate.parameters())
    )
    report = {
        "schema_version": 1,
        "experiment": "aira-neural-babysit-output-adapter-v1",
        "role": (
            "parameter-learning experiment with stored-answer shelf and keyword routes "
            "disabled"
        ),
        "model_sha256": sha256(Path(args.model)),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "curriculum": str(curriculum_path),
        "curriculum_sha256": sha256(curriculum_path),
        "controls": str(controls_path),
        "controls_sha256": sha256(controls_path),
        "training": {
            "steps": config["steps"],
            "records": len(train_rows),
            "supervised_tokens": train.metadata["tokens"],
            "parameters_changed": parameter_count,
            "training_seconds": training_seconds,
            "loss_samples": training_samples,
            "train_teacher_forced": train_metrics,
            "validation_teacher_forced": validation_metrics,
        },
        "gate": gate_results,
        "free_generation": {
            "tasks": len(validation_records),
            "base_concept_passes": sum(
                item["base"]["verification"]["passed"] for item in validation_records
            ),
            "adapted_concept_passes": sum(
                item["adapted"]["verification"]["passed"] for item in validation_records
            ),
            "base_tokens": sum(item["base"]["tokens"] for item in validation_records),
            "adapted_tokens": sum(
                item["adapted"]["tokens"] for item in validation_records
            ),
            "adapted_routes_from_shelf": 0,
            "records": validation_records,
        },
        "out_of_scope_controls": {
            "tasks": len(control_records),
            "adapter_activations": sum(
                item["adapted"]["adapter_active"] for item in control_records
            ),
            "byte_exact_preserved": sum(
                item["byte_exact_preserved"] for item in control_records
            ),
            "records": control_records,
        },
        "timing_seconds": {
            "collection": collection_seconds,
            "generation": generation_seconds,
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "adapter": str(adapter_path),
        "adapter_sha256": sha256(adapter_path),
        "acceptance": {
            "weights_changed": parameter_count > 0,
            "held_out_teacher_forced_nll_improved": (
                validation_metrics["adapted_nll"] < validation_metrics["base_nll"]
            ),
            "held_out_teacher_forced_top1_improved": (
                validation_metrics["adapted_top1_accuracy"]
                > validation_metrics["base_top1_accuracy"]
            ),
            "held_out_free_generation_improved": (
                sum(
                    item["adapted"]["verification"]["passed"]
                    for item in validation_records
                )
                > sum(
                    item["base"]["verification"]["passed"]
                    for item in validation_records
                )
            ),
            "all_positive_prompts_activated": gate_results["positive_validation"][
                "accuracy"
            ]
            == 1.0,
            "all_negative_controls_preserved": all(
                item["byte_exact_preserved"] for item in control_records
            ),
            "production_deployment_allowed": False,
        },
        "limitations": (
            "This is a real learned logit residual, not a stored-answer route, but the "
            "tiny correction set can still overfit. Keyword concept checks need manual "
            "review, and any out-of-scope gate false positive keeps deployment closed."
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (artifact / "validation_generation.jsonl").write_text(
        "".join(
            json.dumps(item, ensure_ascii=False) + "\n" for item in validation_records
        ),
        encoding="utf-8",
    )
    (artifact / "control_generation.jsonl").write_text(
        "".join(
            json.dumps(item, ensure_ascii=False) + "\n" for item in control_records
        ),
        encoding="utf-8",
    )
    (artifact / "README.md").write_text(
        "# AIra Neural Babysit v1\n\n"
        "This artifact contains a learned prompt-gated hidden-to-logit residual, not a "
        "SkillShelf or stored-answer route. The frozen Qwen donor supplies 1024-wide "
        "hidden states; 264,137 adapter parameters were optimized for exactly 300 steps.\n\n"
        f"- Training records: {len(train_rows)}\n"
        f"- Held-out free-generation concept passes: "
        f"{report['free_generation']['base_concept_passes']}/24 -> "
        f"{report['free_generation']['adapted_concept_passes']}/24 before manual review\n"
        f"- Out-of-scope answers preserved: "
        f"{report['out_of_scope_controls']['byte_exact_preserved']}/"
        f"{report['out_of_scope_controls']['tasks']}\n"
        "- Production deployment: blocked\n\n"
        "`adapter.bin` is the native numeric checkpoint; `model.pt` preserves the "
        "PyTorch tensors; TSV/JSONL files bind training, held-out prompts, outputs, and "
        "provenance. Teacher answers are not runtime inputs. See "
        "`docs/aira-neural-babysit-v1.md` and the manually audited result for limits.\n",
        encoding="utf-8",
    )
    evidence_files = [
        path
        for path in artifact.iterdir()
        if path.is_file() and path.name != "manifest.json"
    ]
    manifest = {
        "schema_version": 1,
        "files": [
            {
                "path": str(path.relative_to(artifact)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(evidence_files)
        ],
        "model_sha256": report["model_sha256"],
        "training_steps": config["steps"],
        "stored_answer_routes_used": False,
    }
    (artifact / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "parameters_changed": parameter_count,
                "teacher_forced_validation": validation_metrics,
                "free_generation": {
                    key: value
                    for key, value in report["free_generation"].items()
                    if key != "records"
                },
                "out_of_scope_controls": {
                    key: value
                    for key, value in report["out_of_scope_controls"].items()
                    if key != "records"
                },
                "acceptance": report["acceptance"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
