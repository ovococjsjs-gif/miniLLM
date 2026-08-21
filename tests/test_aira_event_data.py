from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from minillm.aira.event_data import EventDatasetWriter, read_event_shards
from minillm.aira.events import EventTrainingBatch


def example_batch(count: int = 3) -> EventTrainingBatch:
    return EventTrainingBatch(
        contexts=np.arange(count * 4, dtype=np.int64).reshape(count, 4),
        byte_targets=np.arange(count * 2, dtype=np.int64).reshape(count, 2),
        target_lengths=np.full(count, 2, dtype=np.int64),
        route_targets=np.zeros(count, dtype=np.int64),
        byte_supervised=np.ones(count, dtype=bool),
        full_event_lengths=np.full(count, 2, dtype=np.int64),
    )


def test_event_dataset_shards_round_trip_and_record_identity(tmp_path: Path) -> None:
    root = tmp_path / "events"
    writer = EventDatasetWriter(
        root, maximum_events_per_shard=4, metadata={"tokenizer_sha256": "abc"}
    )
    writer.add("train", example_batch(3))
    writer.add("train", example_batch(3))
    writer.add("validation", example_batch(2))
    manifest = writer.close()

    train = list(read_event_shards(root, split="train"))

    assert manifest["events"] == 8
    assert manifest["splits"] == {"train": 6, "validation": 2}
    assert manifest["metadata"] == {"tokenizer_sha256": "abc"}
    assert len(train) == 2
    assert sum(len(batch.contexts) for batch in train) == 6
    np.testing.assert_array_equal(train[0].contexts, example_batch().contexts)


def test_event_dataset_detects_tampered_shard(tmp_path: Path) -> None:
    root = tmp_path / "events"
    writer = EventDatasetWriter(root)
    writer.add("train", example_batch())
    manifest = writer.close()
    path = root / manifest["shards"][0]["path"]
    with path.open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(ValueError, match="hash mismatch"):
        list(read_event_shards(root))
