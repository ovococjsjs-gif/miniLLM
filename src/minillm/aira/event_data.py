"""Deterministic sharded arrays for event-model training and resume identity."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .events import EventTrainingBatch


@dataclass(frozen=True)
class EventShardInfo:
    split: str
    path: str
    events: int
    bytes: int
    sha256: str


class EventDatasetWriter:
    """Write deterministic, bounded NPZ shards without pickle objects."""

    def __init__(
        self,
        root: str | Path,
        *,
        maximum_events_per_shard: int = 100_000,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if maximum_events_per_shard < 1:
            raise ValueError("maximum events per shard must be positive")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if (self.root / "manifest.json").exists():
            raise FileExistsError("event dataset already has a manifest")
        self.maximum_events = maximum_events_per_shard
        self.metadata = dict(metadata or {})
        self.buffers: dict[str, list[EventTrainingBatch]] = defaultdict(list)
        self.buffered_events: dict[str, int] = defaultdict(int)
        self.indices: dict[str, int] = defaultdict(int)
        self.shards: list[EventShardInfo] = []
        self.closed = False

    def add(self, split: str, batch: EventTrainingBatch) -> None:
        if self.closed or not split:
            raise ValueError("writer is closed or split is empty")
        count = len(batch.contexts)
        if count < 1:
            raise ValueError("cannot add an empty event batch")
        if self.buffered_events[split] and (
            self.buffered_events[split] + count > self.maximum_events
        ):
            self._flush(split)
        self.buffers[split].append(batch)
        self.buffered_events[split] += count
        if self.buffered_events[split] >= self.maximum_events:
            self._flush(split)

    def _flush(self, split: str) -> None:
        batches = self.buffers.pop(split, [])
        if not batches:
            return
        directory = self.root / split
        directory.mkdir(parents=True, exist_ok=True)
        index = self.indices[split]
        self.indices[split] += 1
        relative = Path(split) / f"part-{index:05d}.npz"
        output = self.root / relative
        temporary = output.with_suffix(output.suffix + ".tmp")
        arrays = {
            "contexts": np.concatenate([batch.contexts for batch in batches]).astype(
                np.uint32
            ),
            "byte_targets": np.concatenate(
                [batch.byte_targets for batch in batches]
            ).astype(np.uint8),
            "target_lengths": np.concatenate(
                [batch.target_lengths for batch in batches]
            ).astype(np.uint8),
            "route_targets": np.concatenate(
                [batch.route_targets for batch in batches]
            ).astype(np.uint8),
            "byte_supervised": np.concatenate(
                [batch.byte_supervised for batch in batches]
            ).astype(bool),
            "full_event_lengths": np.concatenate(
                [batch.full_event_lengths for batch in batches]
            ).astype(np.uint16),
        }
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
        os.replace(temporary, output)
        with output.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        self.shards.append(
            EventShardInfo(
                split=split,
                path=str(relative),
                events=len(arrays["contexts"]),
                bytes=output.stat().st_size,
                sha256=digest,
            )
        )
        self.buffered_events[split] = 0

    def close(self) -> dict[str, Any]:
        if self.closed:
            raise RuntimeError("event dataset writer is already closed")
        for split in sorted(self.buffers):
            self._flush(split)
        self.closed = True
        manifest = {
            "schema_version": 1,
            "events": sum(shard.events for shard in self.shards),
            "splits": {
                split: sum(
                    shard.events for shard in self.shards if shard.split == split
                )
                for split in sorted({shard.split for shard in self.shards})
            },
            "maximum_events_per_shard": self.maximum_events,
            "metadata": self.metadata,
            "shards": [asdict(shard) for shard in self.shards],
        }
        temporary = self.root / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(temporary, self.root / "manifest.json")
        return manifest


def read_event_shards(
    root: str | Path,
    *,
    split: str | None = None,
    verify_hashes: bool = True,
) -> Iterator[EventTrainingBatch]:
    dataset = Path(root)
    manifest = json.loads((dataset / "manifest.json").read_text())
    for shard in manifest["shards"]:
        if split is not None and shard["split"] != split:
            continue
        path = dataset / shard["path"]
        if verify_hashes:
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != shard["sha256"]:
                raise ValueError(f"event shard hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as payload:
            yield EventTrainingBatch(
                contexts=payload["contexts"].astype(np.int64),
                byte_targets=payload["byte_targets"].astype(np.int64),
                target_lengths=payload["target_lengths"].astype(np.int64),
                route_targets=payload["route_targets"].astype(np.int64),
                byte_supervised=payload["byte_supervised"].astype(bool),
                full_event_lengths=payload["full_event_lengths"].astype(np.int64),
            )
