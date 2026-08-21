from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from minillm.aira.transition_data import EventNormalization, TransitionCorpus


def test_event_normalization_applies_train_statistics() -> None:
    normalization = EventNormalization(
        mean=torch.tensor([1.0, 2.0]),
        scale=torch.tensor([0.5, 2.0]),
        records=3,
    )
    actual = normalization.apply(torch.tensor([[3.0, 2.5]]))
    assert torch.allclose(actual, torch.tensor([[1.0, 1.0]]))
    with pytest.raises(ValueError, match="width"):
        normalization.apply(torch.zeros(1, 3))


def test_transition_corpus_rejects_tampered_array(tmp_path: Path) -> None:
    source = tmp_path / "source"
    raw = tmp_path / "raw"
    source.mkdir()
    raw.mkdir()
    event = np.zeros((1, 2), dtype=np.float32)
    split = np.zeros(1, dtype=np.uint8)
    np.save(source / "event.npy", event)
    np.save(source / "split.npy", split)
    samples = source / "samples.jsonl"
    samples.write_text(json.dumps({"prompt_id": "p", "stage": 1}) + "\n")

    def metadata(path: Path, value: np.ndarray) -> dict:
        return {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    manifest = {
        "arrays": {
            "event_features": metadata(source / "event.npy", event),
            "split": metadata(source / "split.npy", split),
        },
        "samples": {
            "path": samples.name,
            "sha256": hashlib.sha256(samples.read_bytes()).hexdigest(),
        },
    }
    (source / "manifest.json").write_text(json.dumps(manifest))
    with (source / "event.npy").open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(ValueError, match="hash mismatch"):
        TransitionCorpus.load(source, raw)
