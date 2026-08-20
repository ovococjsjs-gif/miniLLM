#!/usr/bin/env python3
"""Calibrate shelf bypass precision on one split and test it on another."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from minillm.aira import (
    build_compact_shelf,
    calibrate_reliability_threshold,
    route_hierarchical_shelf,
)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def char_mapping(text: str) -> dict[str, int]:
    counts = Counter(text)
    symbols = sorted(
        symbol
        for symbol, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
            :60
        ]
    )
    return {symbol: index + 4 for index, symbol in enumerate(symbols)}


def score_routes(
    routes, tokens: np.ndarray, extra_gate: np.ndarray | None = None
) -> dict[str, float | int | None]:
    gate = routes.covered if extra_gate is None else routes.covered & extra_gate
    accepted = int(gate.sum())
    correct = int(np.sum(gate & (routes.predictions == tokens)))
    return {
        "evaluated": len(tokens),
        "accepted": accepted,
        "coverage": accepted / len(tokens),
        "correct": correct,
        "precision": correct / accepted if accepted else None,
    }


def evaluate_representation(
    train: np.ndarray,
    calibration: np.ndarray,
    test: np.ndarray,
    *,
    orders: tuple[int, ...],
    target_precision: float,
) -> dict[str, object]:
    levels = [build_compact_shelf(train, order) for order in orders]
    tiny_threshold = np.nextafter(0.0, 1.0)
    calibration_routes = route_hierarchical_shelf(
        levels,
        calibration,
        minimum_support=5,
        confidence_threshold=tiny_threshold,
        confidence_z=1.96,
        selection="confidence",
    )
    candidates = calibration_routes.covered
    fitted = calibrate_reliability_threshold(
        calibration_routes.lower_confidences[candidates],
        (calibration_routes.predictions == calibration)[candidates],
        target_precision=target_precision,
        confidence_z=1.96,
        minimum_accepted=100,
    )
    test_candidates = route_hierarchical_shelf(
        levels,
        test,
        minimum_support=5,
        confidence_threshold=tiny_threshold,
        confidence_z=1.96,
        selection="confidence",
    )
    calibrated_gate = fitted.accept(test_candidates.lower_confidences)
    fixed_routes = route_hierarchical_shelf(
        levels,
        test,
        minimum_support=5,
        confidence_threshold=target_precision,
        confidence_z=1.96,
        selection="confidence",
    )
    return {
        "orders": orders,
        "packed_shelf_bytes": sum(level.packed_bytes for level in levels),
        "candidate_calibration": score_routes(calibration_routes, calibration),
        "calibration": asdict(fitted),
        "fixed_wilson_threshold_test": score_routes(fixed_routes, test),
        "calibrated_test": score_routes(test_candidates, test, calibrated_gate),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--domain-text", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--max-train-characters", type=int, default=2_000_000)
    parser.add_argument("--max-domain-characters", type=int, default=300_000)
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument("--output", default="results/aira_calibration_proxy.json")
    args = parser.parse_args()
    train_path = Path(args.train)
    domain_path = Path(args.domain_text)
    train_text = train_path.read_text(encoding="utf-8", errors="replace")[
        : args.max_train_characters
    ]
    domain_text = domain_path.read_text(encoding="utf-8", errors="replace")[
        : args.max_domain_characters
    ]
    if len(train_text) < 1000 or len(domain_text) < 2000:
        raise ValueError("calibration proxy needs more source text")
    midpoint = len(domain_text) // 2
    calibration_text, test_text = domain_text[:midpoint], domain_text[midpoint:]
    mapping = char_mapping(train_text)
    encode_chars = lambda text: np.fromiter(
        (mapping.get(symbol, 3) for symbol in text), dtype=np.uint32
    )
    encode_bytes = lambda text: np.frombuffer(
        text.encode("utf-8"), dtype=np.uint8
    ).astype(np.uint32)
    payload = {
        "schema_version": 1,
        "experiment": "aira-v2-held-out-domain-calibration-v1",
        "tag": args.tag,
        "warning": "The calibration split is labeled and domain-specific; final test is separate, but this is not an unlabeled OOD detector.",
        "source": {
            "train": str(train_path),
            "train_sha256": sha256(train_path),
            "domain_text": str(domain_path),
            "domain_sha256": sha256(domain_path),
        },
        "protocol": {
            "shelf_train_characters": len(train_text),
            "calibration_characters": len(calibration_text),
            "test_characters": len(test_text),
            "minimum_support": 5,
            "candidate_score": "per-context Wilson 95% lower bound",
            "selection": "maximum lower bound, longest-order tie break",
            "target_precision": args.target_precision,
            "route_calibration": "largest tied score prefix whose aggregate Wilson 95% precision lower bound reaches target",
        },
        "representations": {
            "char60": evaluate_representation(
                encode_chars(train_text),
                encode_chars(calibration_text),
                encode_chars(test_text),
                orders=(4, 6, 8),
                target_precision=args.target_precision,
            ),
            "utf8-byte": evaluate_representation(
                encode_bytes(train_text),
                encode_bytes(calibration_text),
                encode_bytes(test_text),
                orders=(4, 8, 16),
                target_precision=args.target_precision,
            ),
        },
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
