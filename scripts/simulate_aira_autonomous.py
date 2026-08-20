#!/usr/bin/env python3
"""Stress shelf bypasses under autonomous prefixes with an oracle neural fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from minillm.aira import build_compact_shelf, predict_shelf_next


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def repeats_cycle(
    tokens: list[int], candidate: int, repetitions: int = 3, maximum_period: int = 8
) -> bool:
    sequence = tokens + [candidate]
    for period in range(1, min(maximum_period, len(sequence) // repetitions) + 1):
        suffix = sequence[-period:]
        if all(
            sequence[-repeat * period : -(repeat - 1) * period] == suffix
            for repeat in range(2, repetitions + 1)
        ):
            return True
    return False


def simulate(
    levels,
    validation: np.ndarray,
    *,
    starts: np.ndarray,
    horizon: int,
    controlled: bool,
) -> dict[str, float | int]:
    total = correct = shelf_tokens = shelf_correct = oracle_tokens = rejected = 0
    for start in starts:
        prefix = validation[max(0, start - 64) : start].astype(int).tolist()
        burst = 0
        risk = 0.0
        since_anchor = 0
        for offset in range(horizon):
            target = int(validation[start + offset])
            candidate = predict_shelf_next(
                levels,
                np.asarray(prefix, dtype=np.uint32),
                minimum_support=5,
                confidence_threshold=0.95,
                confidence_z=1.96,
                selection="longest",
            )
            use_shelf = candidate is not None
            if controlled and candidate is not None:
                candidate_risk = 1 - candidate.lower_confidence
                blocked = (
                    burst >= 4
                    or risk + candidate_risk > 0.10
                    or since_anchor >= 8
                    or repeats_cycle(prefix, candidate.token)
                )
                if blocked:
                    use_shelf = False
                    rejected += 1
            if use_shelf:
                assert candidate is not None
                emitted = candidate.token
                shelf_tokens += 1
                shelf_correct += emitted == target
                burst += 1
                risk += 1 - candidate.lower_confidence
            else:
                emitted = target
                oracle_tokens += 1
                burst = 0
                risk = 0.0
                since_anchor = 0
            prefix.append(emitted)
            since_anchor += 1
            total += 1
            correct += emitted == target
    return {
        "sequences": len(starts),
        "tokens": total,
        "accuracy": correct / total,
        "shelf_tokens": shelf_tokens,
        "shelf_fraction": shelf_tokens / total,
        "shelf_precision": shelf_correct / shelf_tokens if shelf_tokens else 0.0,
        "oracle_neural_tokens": oracle_tokens,
        "oracle_neural_fraction": oracle_tokens / total,
        "control_rejections": rejected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--max-train-characters", type=int, default=2_000_000)
    parser.add_argument("--max-validation-characters", type=int, default=300_000)
    parser.add_argument("--sequences", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/aira_autonomous_proxy.json")
    args = parser.parse_args()
    train_path = Path(args.train)
    validation_path = Path(args.validation)
    train_text = train_path.read_text(encoding="utf-8", errors="replace")[
        : args.max_train_characters
    ]
    validation_text = validation_path.read_text(encoding="utf-8", errors="replace")[
        : args.max_validation_characters
    ]
    train = np.frombuffer(train_text.encode("utf-8"), dtype=np.uint8).astype(np.uint32)
    validation = np.frombuffer(validation_text.encode("utf-8"), dtype=np.uint8).astype(
        np.uint32
    )
    if len(validation) <= args.horizon + 64:
        raise ValueError("validation stream is too short")
    levels = [build_compact_shelf(train, order) for order in (4, 8, 16)]
    rng = np.random.default_rng(args.seed)
    population = np.arange(64, len(validation) - args.horizon)
    sequence_count = min(args.sequences, len(population))
    starts = np.sort(rng.choice(population, sequence_count, replace=False))
    payload = {
        "schema_version": 1,
        "experiment": "aira-v2-autonomous-shelf-oracle-fallback-v1",
        "tag": args.tag,
        "warning": "The fallback emits the ground-truth token. This isolates autonomous shelf error propagation and neural-call opportunity; it is not LM quality.",
        "source": {
            "train": str(train_path),
            "train_sha256": sha256(train_path),
            "validation": str(validation_path),
            "validation_sha256": sha256(validation_path),
        },
        "protocol": {
            "representation": "utf8-byte",
            "orders": [4, 8, 16],
            "minimum_support": 5,
            "confidence_gate": "per-context Wilson95 lower bound >=0.95",
            "sequences": sequence_count,
            "horizon": args.horizon,
            "seed": args.seed,
            "controlled": "maximum burst 4, cumulative lower-bound risk 0.10, neural anchor every 8, repeated-cycle guard",
        },
        "packed_shelf_bytes": sum(level.packed_bytes for level in levels),
        "uncontrolled": simulate(
            levels,
            validation,
            starts=starts,
            horizon=args.horizon,
            controlled=False,
        ),
        "controlled": simulate(
            levels,
            validation,
            starts=starts,
            horizon=args.horizon,
            controlled=True,
        ),
    }
    output = Path(args.output)
    if output.exists():
        existing = json.loads(output.read_text())
        reports = existing.setdefault("reports", [])
        reports[:] = [report for report in reports if report["tag"] != args.tag]
        reports.append(payload)
        final = existing
    else:
        final = {"schema_version": 1, "reports": [payload]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
