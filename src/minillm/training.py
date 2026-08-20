"""Reproducible single-device proxy pretraining with atomic checkpoints."""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tokenizers import Tokenizer

from .config import MiniLLMConfig
from .corpus import CorpusDocument
from .model import MiniLLM


@dataclass(frozen=True)
class PackedTokenManifest:
    documents: int
    tokens: int
    tokenizer_vocab_size: int
    tokenizer_path: str
    dtype: str = "uint32"


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 1000
    batch_size: int = 8
    sequence_length: int = 256
    gradient_accumulation: int = 1
    learning_rate: float = 3e-4
    minimum_learning_rate_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    gradient_clip: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 20
    checkpoint_interval: int = 500
    seed: int = 42
    device: str = "cpu"
    min_recurrences: int = 1
    max_recurrences: int | None = None

    def validate(self, model: MiniLLMConfig) -> TrainConfig:
        integers = (
            self.steps,
            self.batch_size,
            self.sequence_length,
            self.gradient_accumulation,
            self.eval_interval,
            self.eval_batches,
        )
        if any(value < 1 for value in integers):
            raise ValueError("training counts must be positive")
        if self.sequence_length > model.max_seq_len:
            raise ValueError("training sequence exceeds model context")
        maximum = self.max_recurrences or model.core_repetitions
        if not 1 <= self.min_recurrences <= maximum <= model.max_core_repetitions:
            raise ValueError("invalid recurrence training range")
        return self


class BinaryTokenDataset:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.tokens = np.memmap(self.path, mode="r", dtype=np.uint32)

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    def batch(
        self,
        batch_size: int,
        sequence_length: int,
        *,
        generator: np.random.Generator,
        device: str | torch.device,
    ) -> torch.Tensor:
        maximum = len(self) - sequence_length
        if maximum <= 0:
            raise ValueError("token file is shorter than one sequence")
        starts = generator.integers(0, maximum, size=batch_size)
        array = np.stack(
            [
                np.asarray(self.tokens[start : start + sequence_length], dtype=np.int64)
                for start in starts
            ]
        )
        return torch.from_numpy(array).to(device=device, dtype=torch.long)


def pack_documents(
    documents: Sequence[CorpusDocument],
    tokenizer: Tokenizer,
    *,
    tokenizer_path: str | Path,
    output_path: str | Path,
) -> PackedTokenManifest:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise ValueError("tokenizer has no <eos> token")
    tokens = 0
    with output.open("wb") as handle:
        for document in documents:
            ids = tokenizer.encode(document.text).ids + [eos_id]
            np.asarray(ids, dtype=np.uint32).tofile(handle)
            tokens += len(ids)
    manifest = PackedTokenManifest(
        documents=len(documents),
        tokens=tokens,
        tokenizer_vocab_size=tokenizer.get_vocab_size(),
        tokenizer_path=str(tokenizer_path),
    )
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def learning_rate(step: int, config: TrainConfig) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / max(1, config.warmup_steps)
    progress = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    minimum = config.minimum_learning_rate_ratio
    return config.learning_rate * (minimum + (1 - minimum) * cosine)


@torch.no_grad()
def evaluate_metrics(
    model: MiniLLM,
    dataset: BinaryTokenDataset,
    config: TrainConfig,
    *,
    seed: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    generator = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        "loss": [],
        "main_loss": [],
        "mtp_loss": [],
        "router_loss": [],
    }
    for _ in range(config.eval_batches):
        batch = dataset.batch(
            config.batch_size,
            config.sequence_length,
            generator=generator,
            device=config.device,
        )
        output = model(batch, labels=batch)
        assert output.loss is not None
        values["loss"].append(float(output.loss))
        values["main_loss"].append(
            float(output.main_loss) if output.main_loss is not None else 0.0
        )
        values["mtp_loss"].append(
            float(output.mtp_loss) if output.mtp_loss is not None else 0.0
        )
        values["router_loss"].append(
            float(output.router_loss) if output.router_loss is not None else 0.0
        )
    model.train(was_training)
    return {name: sum(items) / len(items) for name, items in values.items()}


def _data_signature(path: str | Path) -> dict[str, Any]:
    """Return a cheap identity check without hashing an entire large token stream."""

    import hashlib

    token_path = Path(path)
    size = token_path.stat().st_size
    sample_size = 4096
    with token_path.open("rb") as handle:
        first = handle.read(sample_size)
        if size > sample_size:
            handle.seek(max(0, size - sample_size))
            last = handle.read(sample_size)
        else:
            last = b""
    digest = hashlib.blake2b(first + last, digest_size=16).hexdigest()
    return {
        "size_bytes": size,
        "modified_ns": token_path.stat().st_mtime_ns,
        "edge_blake2b": digest,
    }


def _truncate_metrics_after(metrics_path: Path, step: int) -> None:
    """Drop stale or partial records after the resumed checkpoint."""

    if not metrics_path.exists():
        return
    retained: list[str] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            break
        if int(record.get("step", -1)) <= step:
            retained.append(json.dumps(record, sort_keys=True))
    temporary = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
    temporary.write_text(
        "\n".join(retained) + ("\n" if retained else ""), encoding="utf-8"
    )
    os.replace(temporary, metrics_path)


def _save_checkpoint(
    path: Path,
    *,
    model: MiniLLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    model_config: MiniLLMConfig,
    train_config: TrainConfig,
    data_generator: np.random.Generator,
    data_signatures: dict[str, dict[str, Any]],
    best_validation: float,
    last_validation: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 2,
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": model_config.to_dict(),
            "train_config": asdict(train_config),
            "data_signatures": data_signatures,
            "best_validation": best_validation,
            "last_validation": last_validation,
            "torch_rng": torch.get_rng_state(),
            "torch_cuda_rng": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "data_generator_state": data_generator.bit_generator.state,
        },
        temporary,
    )
    os.replace(temporary, path)


def train_proxy(
    model_config: MiniLLMConfig,
    train_config: TrainConfig,
    *,
    train_tokens: str | Path,
    validation_tokens: str | Path,
    output_directory: str | Path,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    model_config.validate()
    train_config.validate(model_config)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)
    generator = np.random.default_rng(train_config.seed)
    train_data = BinaryTokenDataset(train_tokens)
    validation_data = BinaryTokenDataset(validation_tokens)
    data_signatures = {
        "train": _data_signature(train_tokens),
        "validation": _data_signature(validation_tokens),
    }
    model = MiniLLM(model_config).to(train_config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=train_config.weight_decay,
    )
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.jsonl"
    maximum_recurrences = train_config.max_recurrences or model_config.core_repetitions
    started = time.perf_counter()
    best_validation = float("inf")
    last_validation = float("nan")
    start_step = 0
    if resume_from is not None:
        checkpoint = torch.load(
            Path(resume_from), map_location=train_config.device, weights_only=False
        )
        if checkpoint.get("format_version") != 2:
            raise ValueError("checkpoint predates resume-safe format version 2")
        if checkpoint["model_config"] != model_config.to_dict():
            raise ValueError("checkpoint model configuration does not match")
        if checkpoint["train_config"] != asdict(train_config):
            raise ValueError("checkpoint training configuration does not match")
        if checkpoint["data_signatures"] != data_signatures:
            raise ValueError("checkpoint token data does not match")
        start_step = int(checkpoint["step"])
        if not 0 <= start_step <= train_config.steps:
            raise ValueError("checkpoint step is outside the requested training run")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if torch.cuda.is_available() and checkpoint["torch_cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["torch_cuda_rng"]]
            )
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        generator.bit_generator.state = checkpoint["data_generator_state"]
        best_validation = float(checkpoint["best_validation"])
        last_validation = float(checkpoint["last_validation"])
        _truncate_metrics_after(metrics_path, start_step)

    metrics_mode = "a" if resume_from is not None else "w"
    with metrics_path.open(metrics_mode, encoding="utf-8") as metrics:
        for step in range(start_step, train_config.steps):
            optimizer.zero_grad(set_to_none=True)
            accumulated = {
                "loss": 0.0,
                "main_loss": 0.0,
                "mtp_loss": 0.0,
                "router_loss": 0.0,
            }
            recurrence = random.randint(
                train_config.min_recurrences, maximum_recurrences
            )
            for _ in range(train_config.gradient_accumulation):
                batch = train_data.batch(
                    train_config.batch_size,
                    train_config.sequence_length,
                    generator=generator,
                    device=train_config.device,
                )
                result = model(batch, labels=batch, core_repetitions=recurrence)
                assert result.loss is not None
                (result.loss / train_config.gradient_accumulation).backward()
                divisor = train_config.gradient_accumulation
                accumulated["loss"] += float(result.loss.detach()) / divisor
                accumulated["main_loss"] += (
                    float(result.main_loss.detach()) / divisor
                    if result.main_loss is not None
                    else 0.0
                )
                accumulated["mtp_loss"] += (
                    float(result.mtp_loss.detach()) / divisor
                    if result.mtp_loss is not None
                    else 0.0
                )
                accumulated["router_loss"] += (
                    float(result.router_loss.detach()) / divisor
                    if result.router_loss is not None
                    else 0.0
                )
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config.gradient_clip
            )
            rate = learning_rate(step, train_config)
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.step()

            record: dict[str, Any] = {
                "step": step + 1,
                "train_loss": accumulated["loss"],
                "train_main_loss": accumulated["main_loss"],
                "train_mtp_loss": accumulated["mtp_loss"],
                "train_router_loss": accumulated["router_loss"],
                "learning_rate": rate,
                "gradient_norm": float(gradient_norm),
                "recurrences": recurrence,
                "tokens_seen": (step + 1)
                * train_config.batch_size
                * train_config.sequence_length
                * train_config.gradient_accumulation,
                "wall_seconds": time.perf_counter() - started,
            }
            if (step + 1) % train_config.eval_interval == 0 or step == 0:
                validation = evaluate_metrics(
                    model,
                    validation_data,
                    train_config,
                    seed=train_config.seed + 100_000,
                )
                record.update(
                    {f"validation_{name}": value for name, value in validation.items()}
                )
                last_validation = validation["main_loss"]
                if last_validation < best_validation:
                    best_validation = last_validation
                    _save_checkpoint(
                        output / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        step=step + 1,
                        model_config=model_config,
                        train_config=train_config,
                        data_generator=generator,
                        data_signatures=data_signatures,
                        best_validation=best_validation,
                        last_validation=last_validation,
                    )
            metrics.write(json.dumps(record, sort_keys=True) + "\n")
            metrics.flush()
            if train_config.checkpoint_interval and (
                (step + 1) % train_config.checkpoint_interval == 0
            ):
                _save_checkpoint(
                    output / f"step-{step + 1}.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step + 1,
                    model_config=model_config,
                    train_config=train_config,
                    data_generator=generator,
                    data_signatures=data_signatures,
                    best_validation=best_validation,
                    last_validation=last_validation,
                )

    summary = {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "steps": train_config.steps,
        "tokens_seen": train_config.steps
        * train_config.batch_size
        * train_config.sequence_length
        * train_config.gradient_accumulation,
        "best_validation_main_loss": best_validation,
        "last_validation_main_loss": last_validation,
        "wall_seconds": time.perf_counter() - started,
        "output_directory": str(output),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
