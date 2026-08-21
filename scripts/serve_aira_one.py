#!/usr/bin/env python3
"""Serve AIra One v0.1 through a local OpenAI-compatible HTTP API."""

from __future__ import annotations

import argparse
import json
import signal
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from minillm.aira import (
    AIraBabysitJournal,
    AIraMode,
    AIraOne,
    LocalDonorRuntime,
    OpenAIChatProvider,
)
from minillm.memory import EpisodicMemoryStore
from minillm.system.documents import DocumentStore


def make_handler(assistant: AIraOne, journal: AIraBabysitJournal):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AIraOne/0.1"

        def _json(self, status: int, payload: Any) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 2_000_000:
                raise ValueError("request body size is invalid")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise TypeError("request body must be an object")
            return payload

        def do_GET(self) -> None:
            if self.path == "/health":
                self._json(200, {"status": "ok", "model": "aira-one-v0.1"})
                return
            if self.path == "/v1/models":
                self._json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": "aira-one-v0.1",
                                "object": "model",
                                "owned_by": "miniLLM-AIra",
                            }
                        ],
                    },
                )
                return
            if self.path == "/aira/stats":
                self._json(200, assistant.stats.to_dict())
                return
            self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            try:
                payload = self._body()
                if self.path == "/aira/feedback":
                    journal.feedback(
                        str(payload["interaction_id"]),
                        verdict=str(payload["verdict"]),
                        correction=str(payload.get("correction", "")),
                        note=str(payload.get("note", "")),
                    )
                    self._json(200, {"ok": True})
                    return
                if self.path != "/v1/chat/completions":
                    self._json(404, {"error": "not found"})
                    return
                if payload.get("stream"):
                    raise ValueError("streaming is not implemented in v0.1")
                messages = payload.get("messages")
                if not isinstance(messages, list) or not messages:
                    raise ValueError("messages must be a non-empty array")
                normalized = [
                    {"role": str(item["role"]), "content": str(item["content"])}
                    for item in messages
                    if isinstance(item, dict)
                    and item.get("role") in {"system", "user", "assistant"}
                ]
                user_indices = [
                    index
                    for index, item in enumerate(normalized)
                    if item["role"] == "user"
                ]
                if not user_indices:
                    raise ValueError("messages need a user turn")
                last_user = user_indices[-1]
                user_text = normalized[last_user]["content"]
                system_text = "\n".join(
                    item["content"]
                    for item in normalized[: last_user + 1]
                    if item["role"] == "system"
                )
                history = [
                    item
                    for item in normalized[:last_user]
                    if item["role"] in {"user", "assistant"}
                ]
                mode = AIraMode(str(payload.get("aira_mode", "balanced")))
                response = assistant.answer(
                    user_text,
                    mode=mode,
                    history=history,
                    system_text=system_text,
                )
                now = int(time.time())
                self._json(
                    200,
                    {
                        "id": response.interaction_id,
                        "object": "chat.completion",
                        "created": now,
                        "model": "aira-one-v0.1",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": response.answer,
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": None,
                            "completion_tokens": None,
                            "total_tokens": None,
                        },
                        "aira": response.to_dict(),
                    },
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self._json(400, {"error": str(error)})
            except (OSError, RuntimeError) as error:
                self._json(500, {"error": f"runtime error: {type(error).__name__}"})

        def log_message(self, format: str, *args: object) -> None:
            print(f"AIra API: {self.address_string()} - {format % args}")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--donor-port", type=int, default=8081)
    parser.add_argument(
        "--model",
        default=(
            "data/external/qwen3.5-0.8b/"
            "Qwen3.5-0.8B-Q4_K_M-github.gguf"
        ),
    )
    parser.add_argument("--memory", default=".aira-one/memory.sqlite")
    parser.add_argument("--documents", default=".aira-one/documents.sqlite")
    parser.add_argument("--journal", default=".aira-one/babysit.jsonl")
    args = parser.parse_args()

    runtime = LocalDonorRuntime(model=args.model, port=args.donor_port)
    memory = EpisodicMemoryStore(args.memory)
    documents = DocumentStore(args.documents)
    journal = AIraBabysitJournal(args.journal)
    provider = OpenAIChatProvider(
        base_url=runtime.endpoint,
        model="aira-one-donor",
        timeout_seconds=240,
    )
    assistant = AIraOne(
        provider,
        memory=memory,
        documents=documents,
        journal=journal,
    )
    server = None

    def stop(*_: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        # Bind the public port before starting the donor so a bind failure cannot
        # orphan a large llama-server child.
        server = HTTPServer((args.host, args.port), make_handler(assistant, journal))
        runtime.start()
        print(f"AIra One API listening on http://{args.host}:{args.port}")
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.server_close()
        runtime.stop()
        memory.close()
        documents.close()


if __name__ == "__main__":
    main()
