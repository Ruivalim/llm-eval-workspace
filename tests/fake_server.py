"""httpx.MockTransport that speaks enough SSE to exercise the client.
Each call to /chat/completions pops the next scripted response."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

import httpx


def sse(chunks: list[dict[str, Any]], usage: dict | None = None, *, raw_tail: str = "") -> str:
    """Build an SSE body from delta dicts. usage chunk goes last (OpenAI style)."""
    out = []
    for delta in chunks:
        finish = delta.pop("finish_reason", None)
        out.append(
            "data: "
            + json.dumps(
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "m",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
            )
        )
    if usage is not None:
        out.append(
            "data: "
            + json.dumps(
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "m",
                    "choices": [],
                    "usage": usage,
                }
            )
        )
    body = "\n\n".join(out) + "\n\n" + raw_tail
    if not raw_tail:
        body += "data: [DONE]\n\n"
    return body


def tool_call_chunks(name: str, arguments: dict, call_id: str = "call_1") -> list[dict]:
    """Argument JSON split over two chunks, like real servers do."""
    args = json.dumps(arguments)
    half = len(args) // 2
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "index": 0,
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": args[:half]},
                }
            ],
        },
        {"tool_calls": [{"index": 0, "function": {"arguments": args[half:]}}]},
        {"finish_reason": "tool_calls"},
    ]


class FakeServer:
    """Scripted responses. A script item is a str (SSE body), an httpx.Response
    or a callable(request) -> Response."""

    def __init__(self, script: list[Any]):
        self.script = list(script)
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.script:
            return httpx.Response(500, json={"error": {"message": "script exhausted"}})
        item = self.script.pop(0)
        if callable(item):
            return item(request)
        if isinstance(item, httpx.Response):
            return item
        return httpx.Response(200, content=item, headers={"content-type": "text/event-stream"})

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))

    def body(self, i: int = -1) -> dict:
        return json.loads(self.requests[i].content)


def slow(seconds: float) -> Callable[[httpx.Request], httpx.Response]:
    def _h(request: httpx.Request) -> httpx.Response:
        time.sleep(seconds)
        raise httpx.ReadTimeout("simulated", request=request)

    return _h
