import json
from pathlib import Path

import numpy as np
import torch

from minillm.config import MiniLLMConfig
from minillm.training import BinaryTokenDataset, TrainConfig, learning_rate, train_proxy


def test_binary_dataset_batch_is_seeded(tmp_path: Path) -> None:
    path = tmp_path / "tokens.bin"
    np.arange(1000, dtype=np.uint32).tofile(path)
    dataset = BinaryTokenDataset(path)
    first = dataset.batch(
        3,
        16,
        generator=np.random.default_rng(1),
        device="cpu",
    )
    second = dataset.batch(
        3,
        16,
        generator=np.random.default_rng(1),
        device="cpu",
    )
    torch.testing.assert_close(first, second)
    assert first.shape == (3, 16)


def test_learning_rate_warmup_and_decay() -> None:
    config = TrainConfig(steps=100, warmup_steps=10, learning_rate=1e-3)
    model = MiniLLMConfig().validate()
    config.validate(model)
    assert learning_rate(0, config) < learning_rate(9, config)
    assert learning_rate(10, config) > learning_rate(99, config)
    assert learning_rate(99, config) >= 1e-4


def test_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    tokens = tmp_path / "tokens.bin"
    (np.arange(256, dtype=np.uint32) % 16).tofile(tokens)
    model_config = MiniLLMConfig(
        vocab_size=16,
        d_model=8,
        n_heads=1,
        n_kv_heads=1,
        head_dim=8,
        ffn_hidden=16,
        max_seq_len=8,
        prelude_layers=(),
        core_layers=("conv",),
        coda_layers=(),
        core_repetitions=1,
        max_core_repetitions=1,
        recurrent_input_injection=False,
        mtp_depth=0,
    ).validate()
    train_config = TrainConfig(
        steps=4,
        batch_size=2,
        sequence_length=8,
        learning_rate=1e-3,
        warmup_steps=1,
        eval_interval=2,
        eval_batches=1,
        checkpoint_interval=2,
        seed=123,
    )
    original = tmp_path / "original"
    resumed = tmp_path / "resumed"
    train_proxy(
        model_config,
        train_config,
        train_tokens=tokens,
        validation_tokens=tokens,
        output_directory=original,
    )
    train_proxy(
        model_config,
        train_config,
        train_tokens=tokens,
        validation_tokens=tokens,
        output_directory=resumed,
        resume_from=original / "step-2.pt",
    )
    full_state = torch.load(original / "step-4.pt", weights_only=False)
    resumed_state = torch.load(resumed / "step-4.pt", weights_only=False)
    assert full_state["step"] == resumed_state["step"] == 4
    for name, tensor in full_state["model"].items():
        assert torch.equal(tensor, resumed_state["model"][name]), name
    assert full_state["data_generator_state"] == resumed_state["data_generator_state"]

    # Resuming in place truncates stale post-checkpoint records instead of duplicating them.
    train_proxy(
        model_config,
        train_config,
        train_tokens=tokens,
        validation_tokens=tokens,
        output_directory=original,
        resume_from=original / "step-2.pt",
    )
    metric_steps = [
        json.loads(line)["step"]
        for line in (original / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert metric_steps == [1, 2, 3, 4]
    rerun_state = torch.load(original / "step-4.pt", weights_only=False)
    for name, tensor in full_state["model"].items():
        assert torch.equal(tensor, rerun_state["model"][name]), name
