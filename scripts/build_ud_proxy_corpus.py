#!/usr/bin/env python3
"""Convert four Universal Dependencies treebanks into tokenizer proxy JSONL."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from minillm.corpus import CorpusBuilder, CorpusDocument, build_manifest, write_jsonl

LANGUAGES = {
    "en": ("UD_English-EWT", "CC-BY-SA-4.0"),
    "ru": ("UD_Russian-SynTagRus", "CC-BY-NC-SA-4.0"),
    "de": ("UD_German-GSD", "CC-BY-SA-4.0"),
    "uk": ("UD_Ukrainian-IU", "CC-BY-NC-SA-4.0"),
}
_TEXT = re.compile(r"^# text = (.+)$", re.MULTILINE)


def source_split(path: Path) -> str:
    name = path.name
    if "-train" in name:
        return "train"
    if "-dev" in name:
        return "validation"
    if "-test" in name:
        return "test"
    return "train"


def parse_treebank(
    root: Path, language: str, acquired: str
) -> dict[str, list[CorpusDocument]]:
    repository, license_name = LANGUAGES[language]
    by_split: dict[str, list[CorpusDocument]] = defaultdict(list)
    for path in sorted((root / language).glob("*.conllu")):
        sentences = _TEXT.findall(path.read_text(encoding="utf-8"))
        split = source_split(path)
        for offset in range(0, len(sentences), 100):
            group = sentences[offset : offset + 100]
            if not group:
                continue
            by_split[split].append(
                CorpusDocument(
                    id=f"ud:{language}:{path.stem}:{offset // 100}",
                    text="\n".join(group),
                    source=f"UniversalDependencies/{repository}",
                    license=license_name,
                    language=language,
                    domain="dependency-treebank-text",
                    acquisition_date=acquired,
                    url=f"https://github.com/UniversalDependencies/{repository}",
                    metadata={"source_file": path.name, "sentences": len(group)},
                )
            )
    return by_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/ud")
    parser.add_argument("--output", default="data/tokenizer-proxy")
    args = parser.parse_args()
    acquired = datetime.now(UTC).date().isoformat()
    all_splits: dict[str, list[CorpusDocument]] = defaultdict(list)
    for language in LANGUAGES:
        parsed = parse_treebank(Path(args.root), language, acquired)
        for split, documents in parsed.items():
            all_splits[split].extend(documents)

    builder = CorpusBuilder(
        allowed_licenses=frozenset({"CC-BY-SA-4.0", "CC-BY-NC-SA-4.0"}),
        min_characters=200,
        reject_pii=True,
    )
    accepted: list[CorpusDocument] = []
    rejections: Counter[str] = Counter()
    output = Path(args.output)
    split_counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        result = builder.build(all_splits[split])
        accepted.extend(result.accepted)
        rejections.update(item.reason for item in result.rejected)
        split_counts[split] = len(result.accepted)
        write_jsonl(output / f"{split}.jsonl", result.accepted)
    manifest = build_manifest(accepted)
    manifest.update(
        {
            "splits": split_counts,
            "rejections": dict(rejections),
            "usage_note": "Research proxy only: Russian and Ukrainian subsets are non-commercial.",
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
