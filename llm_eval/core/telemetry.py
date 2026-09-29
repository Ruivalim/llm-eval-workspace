"""TTFT, total time, token counts and tokens/s. The stream reports the first
token of any kind (reasoning included) as TTFT, mirroring bateria-tui."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text)) if text else 0


@dataclass
class Telemetry:
    ttft_ms: int = 0
    total_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_think: int = 0
    tps: float = 0.0
    usage_from_server: bool = False
    cost_reported: float | None = None  # USD, when the endpoint bills per request


@dataclass
class Stopwatch:
    """Started before the request. Call `first_token()` on the first delta and
    `finish(...)` at the end. Falls back to tiktoken when the server sends no usage."""

    t0: float = field(default_factory=time.perf_counter)
    t_first: float | None = None

    def first_token(self) -> None:
        if self.t_first is None:
            self.t_first = time.perf_counter()

    def finish(
        self,
        prompt_text: str,
        answer: str,
        reasoning: str = "",
        usage: dict | None = None,
    ) -> Telemetry:
        t_end = time.perf_counter()
        total_ms = int((t_end - self.t0) * 1000)
        ttft_ms = int(((self.t_first or t_end) - self.t0) * 1000)
        usage = usage or {}
        tokens_in = int(usage.get("prompt_tokens") or 0) or count_tokens(prompt_text)
        tokens_out = int(usage.get("completion_tokens") or 0) or count_tokens(answer + reasoning)
        details = usage.get("completion_tokens_details") or {}
        tokens_think = int(details.get("reasoning_tokens") or 0) or count_tokens(reasoning)
        decode_s = max(0.001, (t_end - (self.t_first or t_end)))
        return Telemetry(
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tokens_think=tokens_think,
            tps=round(tokens_out / decode_s, 2),
            usage_from_server=bool(usage.get("completion_tokens")),
            cost_reported=usage.get("cost"),
        )
