from pathlib import Path

import numpy as np
import torch

from minillm.config import MiniLLMConfig
from minillm.training import BinaryTokenDataset, TrainConfig, learning_rate


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
