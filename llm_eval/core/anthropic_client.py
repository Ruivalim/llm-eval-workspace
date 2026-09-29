"""Streaming client for the Anthropic Messages API, same contract as
`client.stream_chat`: yields Delta while streaming and one ChatResult at the end,
runs the tool loop through tool_mock, cuts repetition loops, honors CancelHook.

What differs from the OpenAI path, on purpose:
- No temperature or top_p. Current Claude models reject sampling parameters
  with a 400, so the bench runs every Claude model at its defaults. The trace
  records that they were dropped.
- `thinking` is not sent: each model runs with its own default. The bench
  measures the model as it ships; reasoning shows up only when the model
  returns readable thinking text.
- max_tokens is mandatory here. When the prompt sets none, the model's own
  output cap comes from the Models API, bounded by DEFAULT_MAX_TOKENS.
"""

from __future__ import annotations

import contextlib
import json
import threading
from collections.abc import Iterator
from dataclasses import asdict
from typing import Any

import anthropic
import httpx2
from anthropic import Anthropic

from llm_eval.core import tool_mock
from llm_eval.core.client import (
    MAX_TOOL_ROUNDS,
    CancelHook,
    ChatResult,
    Delta,
    GenerationParams,
    LLMError,
    Stopped,
    _LoopWatch,
    open_with_retry,
)
from llm_eval.core.telemetry import Stopwatch
from llm_eval.core.trace import NullTrace, TraceLike

DEFAULT_BASE_URL = "https://api.anthropic.com"
DEFAULT_MAX_TOKENS = 64_000

# Anthropic stop_reason -> the OpenAI finish_reason the engine already knows.
_FINISH = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    # Gateways that speak this API add their own. Xiaomi MiMo cuts repetition
    # server-side (same thing as our loop guard) and filters content.
    "repetition_truncation": "loop",
    "content_filter": "refusal",
}

_output_caps: dict[tuple[str, str], int] = {}
_caps_lock = threading.Lock()


def normalize_base_url(base_url: str) -> str:
    """The SDK appends /v1/messages itself. People paste the root, the /v1, or
    the full messages URL from a gateway's docs; accept all three."""
    url = (base_url or DEFAULT_BASE_URL).rstrip("/")
    for tail in ("/messages", "/v1"):
        if url.endswith(tail):
            url = url[: -len(tail)]
    return url


def is_official(base_url: str) -> bool:
    return httpx2.URL(normalize_base_url(base_url)).host == "api.anthropic.com"


def make_client(
    base_url: str,
    api_key: str | None,
    timeout_s: float,
    http_client: httpx2.Client | None = None,
) -> Anthropic:
    key = api_key or "none"
    # Anthropic reads x-api-key. Compatible gateways (Xiaomi MiMo, others)
    # document Authorization: Bearer instead, so off the official host the key
    # goes in both; each server reads the one it knows.
    headers = None if is_official(base_url) else {"Authorization": f"Bearer {key}"}
    return Anthropic(
        base_url=normalize_base_url(base_url),
        # An explicit key only: the SDK would otherwise fall back to whatever
        # ANTHROPIC_API_KEY or login profile this machine has, and spend it.
        api_key=key,
        timeout=timeout_s,
        max_retries=0,  # open_with_retry owns retries, so they land in the trace
        http_client=http_client,
        default_headers=headers,
    )


def map_error(exc: Exception) -> LLMError:
    """One readable line per failure class, like client.map_error."""
    if isinstance(exc, LLMError):
        return exc
    if isinstance(exc, anthropic.APITimeoutError):
        return LLMError("timeout: the server did not answer in time")
    if isinstance(exc, anthropic.APIConnectionError):
        return LLMError(f"connection failed: {exc}")
    if isinstance(exc, anthropic.AuthenticationError):
        return LLMError("authentication failed (401): check the API key")
    if isinstance(exc, anthropic.NotFoundError):
        return LLMError(f"not found (404): model or route missing: {_body_message(exc)}")
    if isinstance(exc, anthropic.APIStatusError):
        return LLMError(f"HTTP {exc.status_code}: {_body_message(exc)}")
    if isinstance(exc, httpx2.TimeoutException):
        return LLMError("timeout: the server did not answer in time")
    if isinstance(exc, httpx2.HTTPError):
        return LLMError(f"transport error: {exc}")
    return LLMError(f"{type(exc).__name__}: {exc}")


def _body_message(exc: anthropic.APIStatusError) -> str:
    body = exc.body
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            return " ".join(str(err["message"]).split())
    text = " ".join((exc.message or str(exc)).split())
    # Gateways answer an unknown route with their web server's HTML 404 page.
    if "<html" in text.lower():
        return "the server answered with an HTML page, not the API"
    return text[:200]


def _gateway_root(base_url: Any) -> str | None:
    """The origin of a gateway serving this API under a path (…/anthropic),
    or None for the official host and for gateways already at the root."""
    url = httpx2.URL(str(base_url))
    if url.host == "api.anthropic.com" or url.path.strip("/") == "":
        return None
    return f"{url.scheme}://{url.netloc.decode()}"


def list_models(client: Anthropic) -> list[dict[str, Any]]:
    """The Models API. Gateways that mount this API under a path (Xiaomi MiMo,
    DeepSeek, Moonshot at …/anthropic) often answer 404 there and list their
    models in OpenAI shape at the root /v1/models instead, so try that too."""
    try:
        return [{"id": m.id} for m in client.models.list()]
    except anthropic.NotFoundError as exc:
        root = _gateway_root(client.base_url)
        if root is None:
            raise map_error(exc) from exc
        try:
            body = client.with_options(base_url=root).get("/v1/models", cast_to=object)
        except Exception:  # noqa: BLE001 - the original 404 is the better message
            raise map_error(exc) from exc
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list):
            raise map_error(exc) from exc
        return [{"id": m["id"]} for m in data if isinstance(m, dict) and m.get("id")]
    except Exception as exc:  # noqa: BLE001
        raise map_error(exc) from exc


def _plain(value: Any) -> Any:
    """SDK content blocks as plain dicts, so the trace is JSON and not reprs."""
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def output_cap(client: Anthropic, model: str) -> int:
    """The model's max output, from the Models API, cached per process. A
    failed lookup falls back to DEFAULT_MAX_TOKENS: if that is too high for
    the model, the 400 that follows names the limit."""
    key = (str(client.base_url), model)
    with _caps_lock:
        if key in _output_caps:
            return _output_caps[key]
    try:
        cap = client.models.retrieve(model).max_tokens or DEFAULT_MAX_TOKENS
    except Exception:  # noqa: BLE001 - best effort, see docstring
        return DEFAULT_MAX_TOKENS
    cap = min(int(cap), DEFAULT_MAX_TOKENS)
    with _caps_lock:
        _output_caps[key] = cap
    return cap


def anthropic_effort(effort: str) -> dict[str, Any]:
    """Request fields for `effort`. Levels go in output_config.effort; "off" is
    thinking disabled, which older Claude models and gateways like Xiaomi MiMo
    take, and Claude Opus 5.5 / Sonnet 5.5 refuse with a 400 (their error then
    shows as the run's error)."""
    if effort == "default":
        return {}
    if effort == "off":
        return {"thinking": {"type": "disabled"}}
    return {"output_config": {"effort": effort}}


def to_anthropic(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """OpenAI-shaped chat messages -> (system, messages). Only the shapes the
    engine builds: system and user text."""
    system = "\n\n".join(str(m["content"]) for m in messages if m["role"] == "system")
    rest = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
    return system, rest


def to_anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """tool_mock schemas are OpenAI function specs; Anthropic wants input_schema."""
    out = []
    for t in tools:
        fn = t.get("function", t)
        out.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return out


def _usage_dict(usage: Any) -> dict[str, Any]:
    """Anthropic usage -> the OpenAI shape Stopwatch.finish reads. Cache reads
    and writes are still input the model processed, so they count as input."""
    if usage is None:
        return {}
    tokens_in = (
        (usage.input_tokens or 0)
        + (usage.cache_creation_input_tokens or 0)
        + (usage.cache_read_input_tokens or 0)
    )
    out: dict[str, Any] = {
        "prompt_tokens": tokens_in,
        "completion_tokens": usage.output_tokens or 0,
        "total_tokens": tokens_in + (usage.output_tokens or 0),
    }
    details = getattr(usage, "output_tokens_details", None)
    thinking = getattr(details, "thinking_tokens", None) if details else None
    if thinking:
        out["completion_tokens_details"] = {"reasoning_tokens": thinking}
    return out


def _stream_once(
    client: Anthropic,
    kwargs: dict[str, Any],
    sw: Stopwatch,
    trace: list[Delta],
    log: TraceLike,
    round_no: int,
    hook: CancelHook | None,
) -> Iterator[Delta | tuple[Any, str | None]]:
    """One request. Yields Delta, then (final message or None, finish_reason)."""

    def _open() -> Any:
        # Entering the manager sends the request, so HTTP errors surface here
        # and open_with_retry can retry them before any output exists.
        return client.messages.stream(**kwargs).__enter__()

    stream = open_with_retry(_open, map_error, log, round_no, hook)
    content_watch, reasoning_watch = _LoopWatch(), _LoopWatch()
    looped: str | None = None
    if hook is not None:
        hook.register(stream)
    try:
        for event in stream:
            if hook is not None and hook.cancelled:
                with contextlib.suppress(Exception):
                    stream.close()
                raise Stopped("stopped by the client")
            if event.type not in ("content_block_delta", "message_start", "message_delta"):
                continue
            log.event("chunk", round=round_no, raw=event.model_dump(exclude_none=True))
            if event.type != "content_block_delta":
                continue
            delta = event.delta
            if delta.type == "thinking_delta" and delta.thinking:
                sw.first_token()
                d = Delta("reasoning", delta.thinking)
                trace.append(d)
                yield d
                if reasoning_watch.feed(delta.thinking):
                    looped = "reasoning"
                    break
            elif delta.type == "text_delta" and delta.text:
                sw.first_token()
                d = Delta("content", delta.text)
                trace.append(d)
                yield d
                if content_watch.feed(delta.text):
                    looped = "content"
                    break
            elif delta.type == "input_json_delta":
                sw.first_token()
    except Stopped:
        raise
    except Exception as exc:  # noqa: BLE001
        if hook is not None and hook.cancelled:
            raise Stopped("stopped by the client") from exc
        err = map_error(exc)
        log.event("error", round=round_no, where="stream", error=str(err), cause=repr(exc))
        raise err from exc
    finally:
        if hook is not None:
            hook.unregister(stream)

    if looped is not None:
        log.event("loop_cut", round=round_no, where=looped)
        message = stream.current_message_snapshot
        with contextlib.suppress(Exception):
            stream.close()
        yield message, "loop"
        return
    message = stream.get_final_message()
    with contextlib.suppress(Exception):
        stream.close()
    finish = _FINISH.get(message.stop_reason or "", message.stop_reason)
    log.event(
        "round_end",
        round=round_no,
        finish_reason=finish,
        stop_reason=message.stop_reason,
        stop_details=getattr(message, "stop_details", None),
        usage=message.usage.model_dump(exclude_none=True) if message.usage else None,
        content=[b.model_dump(exclude_none=True) for b in message.content],
    )
    yield message, finish


def stream_messages(
    client: Anthropic,
    model: str,
    messages: list[dict[str, Any]],
    params: GenerationParams,
    tools: list[dict[str, Any]] | None = None,
    log: TraceLike | None = None,
    hook: CancelHook | None = None,
) -> Iterator[Delta | ChatResult]:
    log = log or NullTrace()
    sw = Stopwatch()
    trace: list[Delta] = []
    system, convo = to_anthropic(messages)
    prompt_text = "\n".join(str(m.get("content") or "") for m in messages)
    max_tokens = params.max_tokens or output_cap(client, model)
    kwargs: dict[str, Any] = {"model": model, "max_tokens": max_tokens}
    if system.strip():
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = to_anthropic_tools(tools)
    kwargs.update(anthropic_effort(params.effort))
    all_answer: list[str] = []
    all_reasoning: list[str] = []
    called: list[str] = []
    usage_total: dict[str, Any] = {}
    finish: str | None = None

    for round_no in range(MAX_TOOL_ROUNDS + 1):
        log.event(
            "request",
            round=round_no,
            base_url=str(client.base_url),
            dropped=["temperature", "top_p"],
            messages=_plain(convo),
            **kwargs,
        )
        message = None
        for item in _stream_once(
            client, {**kwargs, "messages": convo}, sw, trace, log, round_no, hook
        ):
            if isinstance(item, Delta):
                yield item
            else:
                message, finish = item
        assert message is not None
        blocks = list(message.content or [])
        all_answer.append("".join(b.text for b in blocks if b.type == "text"))
        all_reasoning.append("".join(b.thinking or "" for b in blocks if b.type == "thinking"))
        _add_usage(usage_total, _usage_dict(message.usage))
        uses = [b for b in blocks if b.type == "tool_use"]
        if finish != "tool_calls" or not uses:
            break
        if round_no == MAX_TOOL_ROUNDS:
            log.event("error", where="tool_loop", error=f"exceeded {MAX_TOOL_ROUNDS} rounds")
            raise LLMError(f"tool loop exceeded {MAX_TOOL_ROUNDS} rounds")
        # The whole assistant turn goes back unchanged, thinking blocks
        # included: current models reject a history with them edited out.
        convo.append({"role": "assistant", "content": blocks})
        results = []
        for b in uses:
            args = b.input if isinstance(b.input, dict) else {}
            called.append(b.name)
            d_call = Delta("tool_call", f"{b.name}({json.dumps(args, ensure_ascii=False)})")
            trace.append(d_call)
            log.event("tool_call", round=round_no, id=b.id, name=b.name, arguments=args)
            yield d_call
            result = tool_mock.execute(b.name, args)
            log.event("tool_result", round=round_no, id=b.id, name=b.name, result=result)
            d_res = Delta("tool_result", result)
            trace.append(d_res)
            yield d_res
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": result})
        # Every result of one turn in a single user message.
        convo.append({"role": "user", "content": results})

    answer = "".join(all_answer)
    reasoning = "".join(all_reasoning)
    telemetry = sw.finish(prompt_text, answer, reasoning, usage=usage_total or None)
    log.event(
        "result",
        finish_reason=finish,
        tool_calls=called,
        telemetry=asdict(telemetry),
        answer=answer,
        reasoning=reasoning,
    )
    yield ChatResult(answer, reasoning, called, telemetry, trace, finish)


def _add_usage(total: dict[str, Any], usage: dict[str, Any]) -> None:
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if usage.get(k):
            total[k] = total.get(k, 0) + int(usage[k])
    think = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    if think:
        sub = total.setdefault("completion_tokens_details", {})
        sub["reasoning_tokens"] = sub.get("reasoning_tokens", 0) + int(think)


__all__ = [
    "DEFAULT_BASE_URL",
    "list_models",
    "make_client",
    "map_error",
    "normalize_base_url",
    "stream_messages",
    "to_anthropic",
    "to_anthropic_tools",
]
