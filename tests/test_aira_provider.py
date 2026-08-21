from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from minillm.aira.provider import OpenAIChatProvider


class ChatHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        assert self.path == "/v1/chat/completions"
        assert payload["messages"][0]["role"] == "user"
        response = json.dumps(
            {
                "model": payload["model"],
                "choices": [
                    {
                        "message": {
                            "content": "4",
                            "reasoning_content": "",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: object) -> None:
        pass


def test_openai_provider_maps_local_chat_schema() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), ChatHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        provider = OpenAIChatProvider(
            base_url=f"http://127.0.0.1:{server.server_port}",
            model="local-donor",
            timeout_seconds=5,
        )
        response = provider.complete([{"role": "user", "content": "2 + 2?"}])
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    assert response.content == "4"
    assert response.finish_reason == "stop"
    assert response.prompt_tokens == 5
    assert response.completion_tokens == 1
    assert response.raw_model == "local-donor"
