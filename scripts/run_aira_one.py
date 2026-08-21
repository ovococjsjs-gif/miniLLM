#!/usr/bin/env python3
"""Run AIra One v0.1 as a local interactive assistant."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

from minillm.aira import (
    AIraBabysitJournal,
    AIraMode,
    AIraOne,
    LocalDonorRuntime,
    OpenAIChatProvider,
)
from minillm.memory import EpisodicMemoryStore
from minillm.system.documents import DocumentStore


def add_document(store: DocumentStore, path: Path) -> int:
    content = path.read_text(encoding="utf-8")
    return store.add_document(
        title=path.name,
        source=str(path.resolve()),
        license="user-provided-local",
        text=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=(
            "data/external/qwen3.5-0.8b/"
            "Qwen3.5-0.8B-Q4_K_M-github.gguf"
        ),
    )
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--mode", choices=[item.value for item in AIraMode], default="balanced")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--memory", default=".aira-one/memory.sqlite")
    parser.add_argument("--documents", default=".aira-one/documents.sqlite")
    parser.add_argument("--journal", default=".aira-one/babysit.jsonl")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    runtime = None
    endpoint = args.endpoint
    if not args.offline and not endpoint:
        runtime = LocalDonorRuntime(model=args.model, port=args.port)
        endpoint = runtime.endpoint
    provider = (
        None
        if args.offline
        else OpenAIChatProvider(
            base_url=endpoint,
            model="aira-one-donor",
            timeout_seconds=240,
        )
    )
    memory = EpisodicMemoryStore(args.memory)
    documents = DocumentStore(args.documents)
    journal = AIraBabysitJournal(args.journal)
    assistant = AIraOne(
        provider,
        memory=memory,
        documents=documents,
        journal=journal,
    )
    mode = AIraMode(args.mode)
    history: list[dict[str, str]] = []
    last_interaction = ""

    context = runtime if runtime is not None else nullcontext()
    try:
        with context:
            if args.prompt:
                response = assistant.answer(args.prompt, mode=mode)
                print(
                    json.dumps(response.to_dict(), ensure_ascii=False, indent=2)
                    if args.json
                    else response.answer
                )
                return

            print("AIra One v0.1 — локальный RU/EN ассистент")
            print("Команды: /mode fast|balanced|deep, /stats, /add-doc FILE,")
            print("         /feedback correct, /feedback wrong CORRECTION, /clear, /quit")
            while True:
                try:
                    user = input(f"\n[{mode.value}] Вы> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not user:
                    continue
                if user in {"/quit", "/exit"}:
                    break
                if user.startswith("/mode "):
                    mode = AIraMode(user.split(maxsplit=1)[1].strip())
                    print(f"Режим: {mode.value}")
                    continue
                if user == "/stats":
                    print(json.dumps(assistant.stats.to_dict(), ensure_ascii=False, indent=2))
                    continue
                if user.startswith("/add-doc "):
                    document_id = add_document(
                        documents, Path(user.split(maxsplit=1)[1].strip())
                    )
                    print(f"Документ добавлен, id={document_id}")
                    continue
                if user == "/feedback correct":
                    if not last_interaction:
                        print("Сначала задайте вопрос.")
                    else:
                        journal.feedback(last_interaction, verdict="correct")
                        print("Спасибо. Положительная оценка записана в AI Babysit.")
                    continue
                if user.startswith("/feedback wrong "):
                    if not last_interaction:
                        print("Сначала задайте вопрос.")
                    else:
                        correction = user.removeprefix("/feedback wrong ").strip()
                        journal.feedback(
                            last_interaction,
                            verdict="incorrect",
                            correction=correction,
                        )
                        print("Исправление записано в AI Babysit.")
                    continue
                if user == "/clear":
                    history.clear()
                    print("История диалога очищена; подтверждённая память сохранена.")
                    continue

                response = assistant.answer(user, mode=mode, history=history)
                last_interaction = response.interaction_id
                print(f"AIra> {response.answer}")
                print(
                    f"  [{response.route}; {response.latency_seconds:.3f}s; "
                    f"neural calls: {response.neural_calls}]"
                )
                history.append({"role": "user", "content": user})
                history.append({"role": "assistant", "content": response.answer})
    finally:
        memory.close()
        documents.close()


if __name__ == "__main__":
    main()
