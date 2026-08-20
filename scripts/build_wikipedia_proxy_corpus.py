#!/usr/bin/env python3
"""Build a small multilingual tokenizer corpus from random Wikipedia articles."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from minillm.corpus import (
    CorpusBuilder,
    CorpusDocument,
    build_manifest,
    stable_split,
    write_jsonl,
)


def fetch_random(language: str, count: int) -> list[dict[str, object]]:
    pages: dict[int, dict[str, object]] = {}
    endpoint = f"https://{language}.wikipedia.org/w/api.php"
    while len(pages) < count:
        params = {
            "action": "query",
            "generator": "random",
            "grnnamespace": 0,
            "grnlimit": min(20, count - len(pages)),
            "prop": "extracts|info",
            "explaintext": 1,
            "exsectionformat": "plain",
            "inprop": "url",
            "format": "json",
            "formatversion": 2,
        }
        request = urllib.request.Request(
            endpoint + "?" + urllib.parse.urlencode(params),
            headers={"User-Agent": "MiniLLM-Lab/0.1 tokenizer research"},
        )
        payload = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    payload = json.load(response)
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == 4:
                    raise
                time.sleep(0.5 * 2**attempt)
        assert payload is not None
        for page in payload.get("query", {}).get("pages", []):
            if page.get("extract"):
                pages[int(page["pageid"])] = page
        time.sleep(0.15)
    return list(pages.values())[:count]


def structured_documents(acquired: str) -> list[CorpusDocument]:
    templates = {
        "en": "User asks for exact arithmetic. Assistant calls calculator with JSON arguments and grounds the final answer in the tool result.",
        "ru": "Пользователь просит точное вычисление. Ассистент вызывает калькулятор с аргументами JSON и основывает ответ на результате инструмента.",
        "de": "Der Benutzer verlangt eine genaue Berechnung. Der Assistent ruft den Rechner mit JSON-Argumenten auf und stützt die Antwort auf das Werkzeugergebnis.",
        "uk": "Користувач просить точне обчислення. Асистент викликає калькулятор з аргументами JSON і спирається на результат інструмента.",
    }
    documents = []
    for language, description in templates.items():
        for index in range(10):
            left, right = 17 + index, 31 + 2 * index
            text = (
                description
                + "\n"
                + json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": f"calculate: {left}*{right}"},
                            {
                                "role": "assistant",
                                "content": {
                                    "type": "tool_call",
                                    "tool": "calculator",
                                    "arguments": {"expression": f"{left}*{right}"},
                                },
                            },
                            {"role": "tool", "content": {"result": str(left * right)}},
                            {
                                "role": "assistant",
                                "content": {
                                    "type": "final",
                                    "content": str(left * right),
                                    "confidence": 1.0,
                                },
                            },
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            documents.append(
                CorpusDocument(
                    id=f"generated:tools:{language}:{index}",
                    text=text,
                    source="minillm-generated-tool-protocol",
                    license="CC0-1.0",
                    language=language,
                    domain="structured-tool-dialogue",
                    acquisition_date=acquired,
                )
            )
    return documents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", default=["en", "ru", "de", "uk"])
    parser.add_argument("--pages-per-language", type=int, default=60)
    parser.add_argument("--output", default="data/tokenizer-proxy")
    args = parser.parse_args()
    acquired = datetime.now(UTC).date().isoformat()
    raw: list[CorpusDocument] = []
    for language in args.languages:
        for page in fetch_random(language, args.pages_per_language):
            raw.append(
                CorpusDocument(
                    id=f"wikipedia:{language}:{page['pageid']}",
                    text=str(page["extract"]),
                    source=f"{language}.wikipedia.org",
                    license="CC-BY-SA",
                    language=language,
                    domain="encyclopedic",
                    acquisition_date=acquired,
                    url=str(page.get("fullurl", "")) or None,
                    metadata={"title": page.get("title")},
                )
            )
    raw.extend(structured_documents(acquired))
    builder = CorpusBuilder(
        allowed_licenses=frozenset({"CC-BY-SA", "CC0-1.0"}),
        min_characters=200,
        reject_pii=True,
    )
    result = builder.build(raw)
    output = Path(args.output)
    split_counts: Counter[str] = Counter()
    for split in ("train", "validation", "test"):
        documents = [item for item in result.accepted if stable_split(item) == split]
        split_counts[split] = len(documents)
        write_jsonl(output / f"{split}.jsonl", documents)
    manifest = build_manifest(result.accepted)
    manifest.update(
        {
            "splits": dict(split_counts),
            "rejections": Counter(item.reason for item in result.rejected),
            "selection": "MediaWiki random generator; document IDs make the resulting snapshot auditable",
        }
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=dict) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
