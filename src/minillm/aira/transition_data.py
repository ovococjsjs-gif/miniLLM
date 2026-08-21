"""Hash-bound transition corpus access shared by active AIra-Qwen trainers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@dataclass(frozen=True)
class EventNormalization:
    mean: torch.Tensor
    scale: torch.Tensor
    records: int

    def apply(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != self.mean.shape[0]:
            raise ValueError("event feature width differs from normalization")
        return (value - self.mean) * self.scale


@dataclass
class TransitionCorpus:
    source_root: Path
    raw_root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    samples: list[dict[str, Any]]
    token_features: np.ndarray
    split: np.ndarray

    @classmethod
    def load(cls, source_root: str | Path, raw_root: str | Path) -> TransitionCorpus:
        source = Path(source_root)
        raw = Path(raw_root)
        manifest_path = source / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        token_features = cls._verified_array(source, manifest, "event_features")
        split = cls._verified_array(source, manifest, "split")
        sample_path = source / manifest["samples"]["path"]
        if sha256(sample_path) != manifest["samples"]["sha256"]:
            raise ValueError("transition sample metadata hash mismatch")
        samples = [
            json.loads(line)
            for line in sample_path.read_text(encoding="utf-8").splitlines()
        ]
        if len(samples) != token_features.shape[0] or len(samples) != split.shape[0]:
            raise ValueError("transition arrays and metadata have different lengths")
        if not raw.is_dir():
            raise FileNotFoundError(f"raw transition directory is absent: {raw}")
        return cls(source, raw, manifest_path, manifest, samples, token_features, split)

    @staticmethod
    def _verified_array(root: Path, manifest: dict[str, Any], name: str) -> np.ndarray:
        metadata = manifest["arrays"][name]
        path = root / metadata["path"]
        if sha256(path) != metadata["sha256"]:
            raise ValueError(f"transition array hash mismatch: {name}")
        value = np.load(path, allow_pickle=False)
        if (
            list(value.shape) != metadata["shape"]
            or str(value.dtype) != metadata["dtype"]
        ):
            raise ValueError(f"transition array schema mismatch: {name}")
        return value

    @property
    def train_indices(self) -> list[int]:
        return np.flatnonzero(self.split == 0).tolist()

    @property
    def validation_indices(self) -> list[int]:
        return np.flatnonzero(self.split == 1).tolist()

    def tensor_path(self, sample_index: int, name: str, layer: int) -> Path:
        sample = self.samples[sample_index]
        return (
            self.raw_root
            / sample["prompt_id"]
            / f"stage-{sample['stage']}.{name}-{layer}.bin"
        )

    def tensor(
        self, sample_index: int, name: str, layer: int, shape: tuple[int, ...]
    ) -> np.ndarray:
        path = self.tensor_path(sample_index, name, layer)
        value = np.fromfile(path, dtype="<f4")
        if value.size != int(np.prod(shape)):
            raise ValueError(f"unexpected raw transition tensor shape: {path}")
        return value.reshape(shape)

    def event(self, sample_index: int, layer: int) -> np.ndarray:
        if layer < 1:
            raise ValueError("candidate recurrent layer needs a preceding anchor")
        anchor = self.tensor(sample_index, "l_out", layer - 1, (1024,))
        return np.concatenate((self.token_features[sample_index], anchor)).astype(
            np.float32
        )

    def event_normalization(
        self, layers: list[int], *, minimum_standard_deviation: float = 0.05
    ) -> EventNormalization:
        if minimum_standard_deviation <= 0:
            raise ValueError("minimum standard deviation must be positive")
        values = np.stack(
            [
                self.event(sample_index, layer)
                for sample_index in self.train_indices
                for layer in layers
            ]
        ).astype(np.float32)
        mean = torch.from_numpy(values.mean(axis=0))
        standard_deviation = torch.from_numpy(values.std(axis=0)).clamp_min(
            minimum_standard_deviation
        )
        return EventNormalization(mean, standard_deviation.reciprocal(), len(values))

    def validate_candidate_inventory(self, layers: list[int]) -> dict[str, int]:
        tensors = 0
        shifted_rows = 0
        for sample_index in range(len(self.samples)):
            for layer in layers:
                for name, shape in (
                    ("state_predelta", (16, 128, 128)),
                    ("new_state", (16, 128, 128)),
                    ("conv_states", (6144, 3)),
                    ("last_conv_states", (6144, 3)),
                    ("k_conv_predelta", (16, 128)),
                    ("v_conv_predelta", (16, 128)),
                    ("gate", (16,)),
                    ("beta_sigmoid", (16,)),
                ):
                    self.tensor(sample_index, name, layer, shape)
                    tensors += 1
                before = self.tensor(sample_index, "conv_states", layer, (6144, 3)).T
                after = self.tensor(
                    sample_index, "last_conv_states", layer, (6144, 3)
                ).T
                shifted_rows += int(np.array_equal(after[0], before[1]))
                shifted_rows += int(np.array_equal(after[1], before[2]))
        return {
            "samples": len(self.samples),
            "layers": len(layers),
            "tensors": tensors,
            "shifted_row_matches": shifted_rows,
            "shifted_row_comparisons": len(self.samples) * len(layers) * 2,
        }
