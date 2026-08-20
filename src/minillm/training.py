"""Reproducible single-device proxy pretraining with atomic checkpoints."""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
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
    sha256: str
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
    precision: str = "fp32"
    gradient_checkpointing: bool = False
    fused_optimizer: bool = False
    schedule_tokens: int | None = None
    warmup_tokens: int | None = None
    stop_on_nonfinite: bool = True
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
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("precision must be fp32, bf16, or fp16")
        device_type = torch.device(self.device).type
        if self.precision == "fp16" and device_type != "cuda":
            raise ValueError("fp16 training requires CUDA")
        if self.fused_optimizer and device_type != "cuda":
            raise ValueError("fused optimizer requires CUDA")
        if self.schedule_tokens is not None and self.schedule_tokens < 1:
            raise ValueError("schedule_tokens must be positive")
        if self.warmup_tokens is not None and self.warmup_tokens < 0:
            raise ValueError("warmup_tokens cannot be negative")
        if (
            self.schedule_tokens is not None
            and self.warmup_tokens is not None
            and self.warmup_tokens >= self.schedule_tokens
        ):
            raise ValueError("warmup_tokens must be smaller than schedule_tokens")
        if self.checkpoint_interval < 0 or self.warmup_steps < 0:
            raise ValueError("checkpoint interval and warmup cannot be negative")
        maximum = self.max_recurrences or model.core_repetitions
        if not 1 <= self.min_recurrences <= maximum <= model.max_core_repetitions:
            raise ValueError("invalid recurrence training range")
        return self

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.sequence_length * self.gradient_accumulation


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


def pack_document_stream(
    documents: Iterable[CorpusDocument],
    tokenizer: Tokenizer,
    *,
    tokenizer_path: str | Path,
    output_path: str | Path,
) -> PackedTokenManifest:
    """Atomically pack an iterable without materializing the corpus in memory."""

    import hashlib

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    manifest_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or temporary.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite token stream {output}")
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise ValueError("tokenizer has no <eos> token")
    tokens = document_count = 0
    digest = hashlib.sha256()
    try:
        with temporary.open("wb") as handle:
            for document in documents:
                ids = tokenizer.encode(document.text).ids + [eos_id]
                encoded = np.asarray(ids, dtype=np.uint32).tobytes()
                handle.write(encoded)
                digest.update(encoded)
                tokens += len(ids)
                document_count += 1
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    manifest = PackedTokenManifest(
        documents=document_count,
        tokens=tokens,
        tokenizer_vocab_size=tokenizer.get_vocab_size(),
        tokenizer_path=str(tokenizer_path),
        sha256=digest.hexdigest(),
    )
    manifest_path.write_text(
        json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def pack_documents(
    documents: Sequence[CorpusDocument],
    tokenizer: Tokenizer,
    *,
    tokenizer_path: str | Path,
    output_path: str | Path,
) -> PackedTokenManifest:
    return pack_document_stream(
        documents,
        tokenizer,
        tokenizer_path=tokenizer_path,
        output_path=output_path,
    )


def learning_rate(step: int, config: TrainConfig) -> float:
    consumed_tokens = (step + 1) * config.tokens_per_step
    warmup_tokens = (
        config.warmup_tokens
        if config.warmup_tokens is not None
        else config.warmup_steps * config.tokens_per_step
    )
    schedule_tokens = (
        config.schedule_tokens
        if config.schedule_tokens is not None
        else config.steps * config.tokens_per_step
    )
    if warmup_tokens and consumed_tokens <= warmup_tokens:
        return config.learning_rate * consumed_tokens / warmup_tokens
    progress = (consumed_tokens - warmup_tokens) / max(
        1, schedule_tokens - warmup_tokens
    )
    cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    minimum = config.minimum_learning_rate_ratio
    return config.learning_rate * (minimum + (1 - minimum) * cosine)


def _autocast_context(config: TrainConfig):
    if config.precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if config.precision == "bf16" else torch.float16
    return torch.autocast(device_type=torch.device(config.device).type, dtype=dtype)


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
        with _autocast_context(config):
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
    """Validate a packed-token manifest or return a cheap legacy identity check."""

    import hashlib

    token_path = Path(path)
    size = token_path.stat().st_size
    sidecar = token_path.with_suffix(token_path.suffix + ".json")
    if sidecar.exists():
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        dtype = str(manifest["dtype"])
        if dtype != "uint32":
            raise ValueError(f"unsupported packed-token dtype {dtype}: {token_path}")
        expected_size = int(manifest["tokens"]) * np.dtype(dtype).itemsize
        if expected_size != size:
            raise ValueError(f"packed-token size disagrees with manifest: {token_path}")
        with token_path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != manifest["sha256"]:
            raise ValueError(f"packed-token hash disagrees with manifest: {token_path}")
        return {
            "size_bytes": size,
            "tokens": int(manifest["tokens"]),
            "dtype": dtype,
            "sha256": digest,
        }
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
    grad_scaler: Any,
    run_metadata: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 3,
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
            "grad_scaler": grad_scaler.state_dict()
            if grad_scaler.is_enabled()
            else None,
            "run_metadata": dict(run_metadata),
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
    run_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    model_config.validate()
    train_config.validate(model_config)
    metadata = dict(run_metadata or {})
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
    model.set_gradient_checkpointing(train_config.gradient_checkpointing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=train_config.weight_decay,
        fused=train_config.fused_optimizer,
    )
    grad_scaler = torch.amp.GradScaler("cuda", enabled=train_config.precision == "fp16")
    device_type = torch.device(train_config.device).type
    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch.device(train_config.device))
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
        checkpoint_version = int(checkpoint.get("format_version", 0))
        if checkpoint_version not in {2, 3}:
            raise ValueError("checkpoint predates resume-safe format version 2")
        if checkpoint["model_config"] != model_config.to_dict():
            raise ValueError("checkpoint model configuration does not match")
        saved_train_config = dict(checkpoint["train_config"])
        defaults = asdict(TrainConfig())
        for key in asdict(train_config):
            if key in defaults:
                saved_train_config.setdefault(key, defaults[key])
        if saved_train_config != asdict(train_config):
            raise ValueError("checkpoint training configuration does not match")
        saved_metadata = dict(checkpoint.get("run_metadata", {}))
        if saved_metadata != metadata:
            raise ValueError("checkpoint run metadata does not match")
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
        if grad_scaler.is_enabled():
            scaler_state = checkpoint.get("grad_scaler")
            if scaler_state is None:
                raise ValueError(
                    "mixed-precision checkpoint lacks gradient scaler state"
                )
            grad_scaler.load_state_dict(scaler_state)
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
                with _autocast_context(train_config):
                    result = model(batch, labels=batch, core_repetitions=recurrence)
                assert result.loss is not None
                if train_config.stop_on_nonfinite and not torch.isfinite(result.loss):
                    raise FloatingPointError(f"non-finite loss at step {step + 1}")
                scaled_loss = result.loss / train_config.gradient_accumulation
                if grad_scaler.is_enabled():
                    grad_scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
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
            if grad_scaler.is_enabled():
                grad_scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config.gradient_clip
            )
            if train_config.stop_on_nonfinite and not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite gradient norm at step {step + 1}")
            rate = learning_rate(step, train_config)
            for group in optimizer.param_groups:
                group["lr"] = rate
            if grad_scaler.is_enabled():
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
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
                "tokens_seen": (step + 1) * train_config.tokens_per_step,
                "wall_seconds": time.perf_counter() - started,
                "tokens_per_second": (
                    (step + 1 - start_step) * train_config.tokens_per_step
                )
                / max(1e-9, time.perf_counter() - started),
                "precision": train_config.precision,
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
                        grad_scaler=grad_scaler,
                        run_metadata=metadata,
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
                    grad_scaler=grad_scaler,
                    run_metadata=metadata,
                )

    wall_seconds = time.perf_counter() - started
    invocation_tokens = (train_config.steps - start_step) * train_config.tokens_per_step
    peak_device_memory = (
        int(torch.cuda.max_memory_allocated(torch.device(train_config.device)))
        if device_type == "cuda"
        else None
    )
    summary = {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "steps": train_config.steps,
        "resumed_from_step": start_step,
        "tokens_seen": train_config.steps * train_config.tokens_per_step,
        "tokens_processed_this_invocation": invocation_tokens,
        "tokens_per_second": invocation_tokens / max(1e-9, wall_seconds),
        "precision": train_config.precision,
        "gradient_checkpointing": train_config.gradient_checkpointing,
        "peak_device_memory_bytes": peak_device_memory,
        "best_validation_main_loss": best_validation,
        "last_validation_main_loss": last_validation,
        "wall_seconds": wall_seconds,
        "output_directory": str(output),
        "run_metadata": metadata,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
