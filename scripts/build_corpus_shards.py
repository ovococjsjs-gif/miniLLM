#!/usr/bin/env python3
"""Apply source policy and write deterministic, disk-deduplicated corpus shards."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

from minillm.corpus import ContaminationIndex, CorpusDocument, read_jsonl
from minillm.corpus_streaming import StreamingCorpusBuilder
from minillm.data_policy import DataPolicy, SourceRegistry


def documents(paths: list[str]) -> Iterator[CorpusDocument]:
    for path in paths:
        yield from read_jsonl(path)


def protected_texts(paths: list[str]) -> Iterator[str]:
    for path_string in paths:
        path = Path(path_string)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, str):
                    yield item
                elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
                    yield item["prompt"]
                else:
                    raise TypeError(f"unsupported protected item in {path}")
        elif isinstance(raw, dict) and isinstance(raw.get("texts"), list):
            yield from (str(item) for item in raw["texts"])
        else:
            raise TypeError(f"unsupported protected-set format: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--registry", default="configs/corpus/source_registry.json")
    parser.add_argument("--policy", default="configs/corpus/policy_production.json")
    parser.add_argument("--protected", nargs="*", default=["eval/bilingual_smoke.json"])
    parser.add_argument("--shard-mib", type=int, default=256)
    parser.add_argument("--min-characters", type=int, default=200)
    parser.add_argument("--near-duplicate-hamming", type=int, default=3)
    parser.add_argument("--contamination-threshold", type=float, default=0.2)
    parser.add_argument("--allow-pii", action="store_true")
    args = parser.parse_args()

    registry = SourceRegistry.load(args.registry)
    policy = DataPolicy.load(args.policy)
    contamination = (
        ContaminationIndex(protected_texts(args.protected)) if args.protected else None
    )
    builder = StreamingCorpusBuilder(
        args.output,
        registry=registry,
        policy=policy,
        max_shard_bytes=args.shard_mib * 1024 * 1024,
        min_characters=args.min_characters,
        reject_pii=not args.allow_pii,
        near_duplicate_hamming=args.near_duplicate_hamming,
        contamination_index=contamination,
        contamination_threshold=args.contamination_threshold,
    )
    manifest = builder.build(documents(args.inputs))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
