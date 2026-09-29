"""httpx2.MockTransport speaking enough of the Messages API SSE to exercise
anthropic_client. Same scripting model as fake_server.FakeServer."""

from __future__ import annotations

import json
from typing import Any

import httpx2


def _ev(kind: str, data: dict[str, Any]) -> str:
    return f"event: {kind}\ndata: {json.dumps({'type': kind, **data})}\n\n"


def text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "pieces": [text[: len(text) // 2], text[len(text) // 2 :]]}


def thinking_block(text: str) -> dict[str, Any]:
    return {"type": "thinking", "pieces": [text], "signature": "sig-1"}


def tool_block(name: str, args: dict[str, Any], block_id: str = "toolu_1") -> dict[str, Any]:
    raw = json.dumps(args)
    return {"type": "tool_use", "id": block_id, "name": name, "pieces": [raw[:3], raw[3:]]}


def messages_sse(
    blocks: list[dict[str, Any]],
    stop_reason: str = "end_turn",
    usage_in: int = 12,
    usage_out: int = 7,
    cache_read: int = 0,
    thinking_tokens: int | None = None,
) -> str:
    """One full streamed message: start, each block with its deltas, stop."""
    out = [
        _ev(
            "message_start",
            {
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-test",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": usage_in,
                        "output_tokens": 1,
                        "cache_read_input_tokens": cache_read,
                        "cache_creation_input_tokens": 0,
                    },
                }
            },
        )
    ]
    for i, b in enumerate(blocks):
        if b["type"] == "text":
            start = {"type": "text", "text": ""}
            deltas = [{"type": "text_delta", "text": p} for p in b["pieces"]]
        elif b["type"] == "thinking":
            start = {"type": "thinking", "thinking": "", "signature": ""}
            deltas = [{"type": "thinking_delta", "thinking": p} for p in b["pieces"]]
            deltas.append({"type": "signature_delta", "signature": b["signature"]})
        else:
            start = {"type": "tool_use", "id": b["id"], "name": b["name"], "input": {}}
            deltas = [{"type": "input_json_delta", "partial_json": p} for p in b["pieces"]]
        out.append(_ev("content_block_start", {"index": i, "content_block": start}))
        for d in deltas:
            out.append(_ev("content_block_delta", {"index": i, "delta": d}))
        out.append(_ev("content_block_stop", {"index": i}))
    usage: dict[str, Any] = {"output_tokens": usage_out}
    if thinking_tokens is not None:
        usage["output_tokens_details"] = {"thinking_tokens": thinking_tokens}
    out.append(
        _ev(
            "message_delta",
            {"delta": {"stop_reason": stop_reason, "stop_sequence": None}, "usage": usage},
        )
    )
    out.append(_ev("message_stop", {}))
    return "".join(out)


def error(status: int, message: str, headers: dict[str, str] | None = None) -> httpx2.Response:
    kind = {401: "authentication_error", 404: "not_found_error", 429: "rate_limit_error"}.get(
        status, "api_error"
    )
    return httpx2.Response(
        status,
        json={"type": "error", "error": {"type": kind, "message": message}},
        headers=headers or {},
    )


class FakeAnthropic:
    """Scripted /v1/messages responses. GET /v1/models/{id} answers from
    `output_caps` without consuming the script."""

    def __init__(self, script: list[Any], output_caps: dict[str, int] | None = None):
        self.script = list(script)
        self.requests: list[httpx2.Request] = []
        self.output_caps = output_caps or {}

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        if request.method == "GET" and request.url.path.startswith("/v1/models"):
            model = request.url.path.rsplit("/", 1)[-1]
            if model == "models":
                return httpx2.Response(
                    200,
                    json={
                        "data": [
                            {
                                "type": "model",
                                "id": m,
                                "display_name": m,
                                "created_at": "2026-01-01T00:00:00Z",
                            }
                            for m in self.output_caps
                        ],
                        "has_more": False,
                        "first_id": None,
                        "last_id": None,
                    },
                )
            if model not in self.output_caps:
                return error(404, f"model: {model}")
            return httpx2.Response(
                200,
                json={
                    "type": "model",
                    "id": model,
                    "display_name": model,
                    "created_at": "2026-01-01T00:00:00Z",
                    "max_tokens": self.output_caps[model],
                },
            )
        self.requests.append(request)
        if not self.script:
            return error(500, "script exhausted")
        item = self.script.pop(0)
        if callable(item):
            return item(request)
        if isinstance(item, httpx2.Response):
            return item
        return httpx2.Response(200, content=item, headers={"content-type": "text/event-stream"})

    def client(self) -> httpx2.Client:
        return httpx2.Client(transport=httpx2.MockTransport(self.handler))

    def body(self, i: int = -1) -> dict:
        return json.loads(self.requests[i].content)
