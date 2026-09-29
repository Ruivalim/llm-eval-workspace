"""Streaming client for any OpenAI-compatible /v1 endpoint. Yields deltas so
the UI can render live; collects reasoning_content separately when present.

The tool loop runs here too: when a stream ends with tool_calls, the calls are
executed locally via tool_mock and the conversation continues (max ROUNDS)."""

from __future__ import annotations

import contextlib
import json
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import httpx
import openai
from openai import OpenAI

from llm_eval.core import tool_mock
from llm_eval.core.telemetry import Stopwatch, Telemetry
from llm_eval.core.trace import NullTrace, TraceLike

MAX_TOOL_ROUNDS = 5

# Hosted endpoints answer 429 under load and 5xx when busy; a local server
# almost never does. Only these statuses are retried, and only before the
# stream starts: a timeout or a refused connection is the answer, not noise.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504, 529})
MAX_RETRIES = 2
RETRY_MAX_WAIT_S = 30.0

# Repetition guard: small local models sometimes loop on one phrase until
# max_tokens. Cutting the stream early saves minutes per run; the partial
# answer is kept and flagged with finish_reason "loop".
_LOOP_CHECK_EVERY = 1024  # new chars between checks
_LOOP_MIN_BLOCK = 24  # shorter repeats are legit (dividers, lists, code)
_LOOP_MIN_SPAN = 384  # repeated chars before calling it a loop
_LOOP_REPEATS = 4  # same block back-to-back this many times
_LOOP_MAX_BLOCK = 512  # longest block worth scanning for
_LOOP_LINES = 8  # same long line this many times in a row
_LOOP_LINE_LEN = 24


class LLMError(RuntimeError):
    """Readable failure for the UI (connection refused, 4xx, timeout)."""


class Stopped(LLMError):
    """The user stopped the run. Not a failure: never saved, never shown as one."""


class CancelHook:
    """Cross-thread stop for one streamed request. The worker registers each raw
    model stream as it opens; the request owner calls cancel(), which sets the
    flag and closes whatever stream is open. Closing unblocks a worker parked
    in a socket read, so the model server sees the disconnect and drops the
    run instead of generating into a dead socket."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._streams: list[Any] = []
        self._lock = threading.Lock()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def register(self, stream: Any) -> None:
        with self._lock:
            self._streams.append(stream)
            if self._event.is_set():
                # cancel() landed before register: close now, the worker
                # may already be past its last flag check.
                with contextlib.suppress(Exception):
                    stream.close()

    def unregister(self, stream: Any) -> None:
        with self._lock, contextlib.suppress(ValueError):
            self._streams.remove(stream)

    def wait(self, seconds: float) -> bool:
        """Sleep up to `seconds`; True when a stop arrived meanwhile."""
        return self._event.wait(seconds)

    def cancel(self) -> None:
        self._event.set()
        with self._lock:
            streams, self._streams = self._streams, []
        for stream in streams:
            with contextlib.suppress(Exception):
                stream.close()


@dataclass
class Delta:
    kind: str  # "content" | "reasoning" | "tool_call" | "tool_result"
    text: str


@dataclass
class ChatResult:
    answer: str
    reasoning: str
    tool_calls: list[str]
    telemetry: Telemetry
    trace: list[Delta] = field(default_factory=list)
    finish_reason: str | None = (
        None  # "length" = cut by max_tokens, "loop" = cut by the repetition guard
    )


@dataclass
class GenerationParams:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int | None = None  # None: omitted from the request, server decides
    timeout_s: float = 600.0
    effort: str = "default"  # one of EFFORTS


# How hard the model thinks. "default" sends nothing, so the model runs as it
# ships; "off" turns thinking off; the rest is a level, passed through as is.
# Each flavor maps these to its own request field; a model that does not take
# one answers with an error, shown as the run's error, never guessed around.
EFFORTS = ("default", "off", "low", "medium", "high", "xhigh", "max")


def make_client(
    base_url: str,
    api_key: str | None,
    timeout_s: float,
    http_client: httpx.Client | None = None,
) -> OpenAI:
    return OpenAI(
        base_url=base_url,
        api_key=api_key or "none",
        timeout=timeout_s,
        max_retries=0,
        http_client=http_client,
    )


def map_error(exc: Exception) -> LLMError:
    """One readable line per failure class. Keeps the original as __cause__."""
    if isinstance(exc, LLMError):
        return exc  # already readable; wrapping would prefix the class name
    if isinstance(exc, openai.APITimeoutError):
        return LLMError("timeout: the server did not answer in time")
    if isinstance(exc, openai.APIConnectionError):
        return LLMError(f"connection failed: {exc}")
    if isinstance(exc, openai.AuthenticationError):
        return LLMError("authentication failed (401): check the API key")
    if isinstance(exc, openai.NotFoundError):
        return LLMError(f"not found (404): model or route missing: {_body_message(exc)}")
    if isinstance(exc, openai.APIStatusError):
        return LLMError(f"HTTP {exc.status_code}: {_body_message(exc)}")
    if isinstance(exc, httpx.TimeoutException):
        return LLMError("timeout: the server did not answer in time")
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code == 401:
            return LLMError("authentication failed (401): check the API key")
        body = " ".join(exc.response.text.split())[:160] or exc.response.reason_phrase
        return LLMError(f"HTTP {code}: {body}")
    if isinstance(exc, httpx.HTTPError):
        return LLMError(f"transport error: {exc}")
    return LLMError(f"{type(exc).__name__}: {exc}")


def _body_message(exc: openai.APIStatusError) -> str:
    body = exc.body
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        if body.get("message"):
            return str(body["message"])
    return exc.message or str(exc)


@dataclass
class _ToolCallAcc:
    id: str = ""
    name: str = ""
    arguments: str = ""


def _looks_looped(text: str) -> bool:
    """True when the tail of text repeats one block or one line.

    Thresholds stay conservative on purpose: a real loop runs for thousands of
    chars, so waiting for ~400 of pure repetition costs nothing and keeps
    legit repeats (dividers, lists, boilerplate) from tripping the guard."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if (
        len(lines) >= _LOOP_LINES
        and len(lines[-1]) >= _LOOP_LINE_LEN
        and len(set(lines[-_LOOP_LINES:])) == 1
    ):
        return True
    if len(text) < _LOOP_MIN_SPAN:
        return False
    window = text[-(_LOOP_REPEATS * _LOOP_MAX_BLOCK) :]
    for p in range(_LOOP_MIN_BLOCK, min(_LOOP_MAX_BLOCK, len(window) // _LOOP_REPEATS) + 1):
        need = max(_LOOP_REPEATS, -(-_LOOP_MIN_SPAN // p))
        if window.endswith(window[-p:] * need):
            return True
    return False


class _LoopWatch:
    """Incremental _looks_looped: joins the buffer only every CHECK_EVERY chars."""

    def __init__(self) -> None:
        self.pieces: list[str] = []
        self.total = 0
        self.checked = 0

    def feed(self, piece: str) -> bool:
        self.pieces.append(piece)
        self.total += len(piece)
        if self.total - self.checked < _LOOP_CHECK_EVERY:
            return False
        self.checked = self.total
        return _looks_looped("".join(self.pieces))


def openai_effort(effort: str) -> dict[str, Any]:
    """Request fields for `effort` on an OpenAI-compatible server. Levels go in
    reasoning_effort (OpenAI's name). "off" uses thinking.type=disabled, the
    switch the hosted APIs that think by default share (Xiaomi MiMo, DeepSeek,
    GLM, Kimi); OpenAI itself has no off switch and rejects it."""
    if effort == "default":
        return {}
    if effort == "off":
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    return {"reasoning_effort": effort}


def retry_delay(exc: Exception, attempt: int) -> float | None:
    """Seconds to wait before retrying `exc`, or None when it is not retryable.
    Honors a numeric Retry-After, capped, else backs off 1s, 2s, 4s..."""
    if getattr(exc, "status_code", None) not in RETRY_STATUSES:
        return None
    response = getattr(exc, "response", None)
    header = response.headers.get("retry-after") if response is not None else None
    try:
        wait = float(header) if header is not None else float(2**attempt)
    except ValueError:
        wait = float(2**attempt)
    return min(max(wait, 0.0), RETRY_MAX_WAIT_S)


def _sleep(seconds: float, hook: CancelHook | None) -> bool:
    """True when the user stopped during the wait."""
    if hook is not None:
        return hook.wait(seconds)
    time.sleep(seconds)
    return False


def open_with_retry(
    open_stream: Callable[[], Any],
    map_exc: Callable[[Exception], LLMError],
    log: TraceLike,
    round_no: int,
    hook: CancelHook | None = None,
) -> Any:
    """Open a model stream, retrying rate limits and 5xx up to MAX_RETRIES.
    Each retry is a trace event, so waiting time inside the TTFT is explainable."""
    attempt = 0
    while True:
        try:
            return open_stream()
        except Exception as exc:  # noqa: BLE001 - mapped to one readable line
            delay = retry_delay(exc, attempt) if attempt < MAX_RETRIES else None
            err = map_exc(exc)
            if delay is None:
                log.event("error", round=round_no, where="create", error=str(err), cause=repr(exc))
                raise err from exc
            attempt += 1
            log.event("retry", round=round_no, attempt=attempt, wait_s=delay, error=str(err))
            if _sleep(delay, hook):
                raise Stopped("stopped by the client") from exc


def _stream_once(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    params: GenerationParams,
    tools: list[dict[str, Any]] | None,
    sw: Stopwatch,
    trace: list[Delta],
    log: TraceLike,
    round_no: int,
    hook: CancelHook | None = None,
) -> Iterator[Delta | tuple[str, str, list[_ToolCallAcc], dict | None, str | None]]:
    """One request. Yields Delta, then a final tuple
    (content, reasoning, tool_calls, usage, finish_reason)."""
    kwargs: dict[str, Any] = dict(
        model=model,
        messages=messages,
        temperature=params.temperature,
        top_p=params.top_p,
        stream=True,
        stream_options={"include_usage": True},
    )
    if params.max_tokens is not None:
        kwargs["max_tokens"] = params.max_tokens
    if tools:
        kwargs["tools"] = tools
    kwargs.update(openai_effort(params.effort))
    log.event("request", round=round_no, base_url=str(client.base_url), **kwargs)
    stream = open_with_retry(
        lambda: client.chat.completions.create(**kwargs), map_error, log, round_no, hook
    )

    content, reasoning, usage, finish = [], [], None, None
    calls: dict[int, _ToolCallAcc] = {}
    content_watch, reasoning_watch = _LoopWatch(), _LoopWatch()
    looped: str | None = None
    if hook is not None:
        hook.register(stream)
    try:
        for chunk in stream:
            if hook is not None and hook.cancelled:
                # Stop asked mid-stream: close our side first so the model
                # server drops the run instead of finishing it unheard.
                with contextlib.suppress(Exception):
                    stream.close()
                raise Stopped("stopped by the client")
            log.event("chunk", round=round_no, raw=chunk.model_dump(exclude_none=True))
            if chunk.usage is not None:
                usage = chunk.usage.model_dump()
            if not chunk.choices:
                continue
            if chunk.choices[0].finish_reason:
                finish = chunk.choices[0].finish_reason
            d = chunk.choices[0].delta
            if d is None:
                continue
            extra = d.model_extra or {}
            think = extra.get("reasoning_content") or extra.get("reasoning")
            if think:
                sw.first_token()
                reasoning.append(think)
                delta = Delta("reasoning", think)
                trace.append(delta)
                yield delta
                if reasoning_watch.feed(think):
                    looped = "reasoning"
                    break
            if d.content:
                sw.first_token()
                content.append(d.content)
                delta = Delta("content", d.content)
                trace.append(delta)
                yield delta
                if content_watch.feed(d.content):
                    looped = "content"
                    break
            for tc in d.tool_calls or []:
                sw.first_token()
                acc = calls.setdefault(tc.index, _ToolCallAcc())
                if tc.id:
                    acc.id = tc.id
                if tc.function:
                    if tc.function.name:
                        acc.name += tc.function.name
                    if tc.function.arguments:
                        acc.arguments += tc.function.arguments
    except Stopped:
        raise
    except Exception as exc:  # noqa: BLE001
        if hook is not None and hook.cancelled:
            # cancel() closed the stream under a blocked read: that is the
            # stop landing, not a server failure.
            raise Stopped("stopped by the client") from exc
        err = map_error(exc)
        log.event("error", round=round_no, where="stream", error=str(err), cause=repr(exc))
        raise err from exc
    finally:
        if hook is not None:
            hook.unregister(stream)
    if looped is not None:
        # Closing the stream cancels the run server-side; the partial text stays.
        log.event("loop_cut", round=round_no, where=looped)
        with contextlib.suppress(Exception):  # best effort cancel
            stream.close()
        finish = "loop"
    log.event(
        "round_end",
        round=round_no,
        finish_reason=finish,
        usage=usage,
        content_chars=sum(map(len, content)),
        reasoning_chars=sum(map(len, reasoning)),
        tool_calls=[{"id": c.id, "name": c.name, "arguments": c.arguments} for c in calls.values()],
    )
    yield "".join(content), "".join(reasoning), [calls[i] for i in sorted(calls)], usage, finish


def stream_chat(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    params: GenerationParams,
    tools: list[dict[str, Any]] | None = None,
    log: TraceLike | None = None,
    hook: CancelHook | None = None,
) -> Iterator[Delta | ChatResult]:
    """Yield Delta while streaming and a final ChatResult. Tool calls are
    executed via tool_mock and the conversation continues until the model
    answers in plain text (max MAX_TOOL_ROUNDS). A repetition loop cuts the
    stream early with finish_reason "loop". `log` gets every raw chunk."""
    log = log or NullTrace()
    sw = Stopwatch()
    trace: list[Delta] = []
    convo = list(messages)
    prompt_text = "\n".join(str(m.get("content") or "") for m in messages)
    all_answer: list[str] = []
    all_reasoning: list[str] = []
    called: list[str] = []
    usage_total: dict[str, int] = {}
    finish_reason: str | None = None

    for _round in range(MAX_TOOL_ROUNDS + 1):
        final = None
        for item in _stream_once(client, model, convo, params, tools, sw, trace, log, _round, hook):
            if isinstance(item, Delta):
                yield item
            else:
                final = item
        assert final is not None
        content, reasoning, calls, usage, finish_reason = final
        all_answer.append(content)
        all_reasoning.append(reasoning)
        _merge_usage(usage_total, usage)
        if not calls:
            break
        if _round == MAX_TOOL_ROUNDS:
            log.event("error", where="tool_loop", error=f"exceeded {MAX_TOOL_ROUNDS} rounds")
            raise LLMError(f"tool loop exceeded {MAX_TOOL_ROUNDS} rounds")
        convo.append(
            {
                "role": "assistant",
                "content": content or None,
                "tool_calls": [
                    {
                        "id": c.id or f"call_{i}",
                        "type": "function",
                        "function": {"name": c.name, "arguments": c.arguments or "{}"},
                    }
                    for i, c in enumerate(calls)
                ],
            }
        )
        for i, c in enumerate(calls):
            try:
                args = json.loads(c.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": c.arguments}
            called.append(c.name)
            d_call = Delta("tool_call", f"{c.name}({json.dumps(args, ensure_ascii=False)})")
            trace.append(d_call)
            log.event("tool_call", round=_round, id=c.id, name=c.name, arguments=args)
            yield d_call
            result = tool_mock.execute(c.name, args if isinstance(args, dict) else {})
            log.event("tool_result", round=_round, id=c.id, name=c.name, result=result)
            d_res = Delta("tool_result", result)
            trace.append(d_res)
            yield d_res
            convo.append({"role": "tool", "tool_call_id": c.id or f"call_{i}", "content": result})

    answer = "".join(all_answer)
    reasoning = "".join(all_reasoning)
    telemetry = sw.finish(prompt_text, answer, reasoning, usage=usage_total or None)
    log.event(
        "result",
        finish_reason=finish_reason,
        tool_calls=called,
        telemetry=asdict(telemetry),
        answer=answer,
        reasoning=reasoning,
    )
    yield ChatResult(answer, reasoning, called, telemetry, trace, finish_reason)


def _merge_usage(total: dict[str, Any], usage: dict | None) -> None:
    if not usage:
        return
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if usage.get(k):
            total[k] = total.get(k, 0) + int(usage[k])
    # OpenRouter and a few other gateways bill per request and say how much in
    # usage.cost (USD). Kept as reported, never recomputed from a price table.
    cost = usage.get("cost")
    if isinstance(cost, int | float) and not isinstance(cost, bool):
        total["cost"] = total.get("cost", 0.0) + float(cost)
    det = usage.get("completion_tokens_details") or {}
    if det.get("reasoning_tokens"):
        sub = total.setdefault("completion_tokens_details", {})
        sub["reasoning_tokens"] = sub.get("reasoning_tokens", 0) + int(det["reasoning_tokens"])


def apply_fault(
    fault: dict[str, Any],
    api_key: str | None,
    model: str,
    messages: list[dict[str, Any]],
    params: GenerationParams,
) -> tuple[str | None, str, list[dict[str, Any]], GenerationParams]:
    """error_test: the request as `fault.kind` breaks it. Provider-neutral, so
    every flavor fails the same way and the verdicts compare."""
    kind = fault.get("kind")
    if kind == "timeout":
        params = replace(params, timeout_s=float(fault.get("seconds", 2)))
    elif kind == "bad_key":
        api_key = "invalid-key-for-fault-injection"
    elif kind == "wrong_model":
        model = fault.get("model", "model-that-does-not-exist")
    elif kind == "huge_prompt":
        filler = "lorem ipsum " * int(fault.get("words", 200_000))
        messages = [*messages[:-1], {**messages[-1], "content": filler + messages[-1]["content"]}]
    else:
        raise LLMError(f"unknown fault kind {kind!r}")
    return api_key, model, messages, params


def run_with_fault(
    fault: dict[str, Any],
    base_url: str,
    api_key: str | None,
    model: str,
    messages: list[dict[str, Any]],
    params: GenerationParams,
    http_client: httpx.Client | None = None,
    log: TraceLike | None = None,
    hook: CancelHook | None = None,
) -> Iterator[Delta | ChatResult]:
    """error_test: apply `fault.kind` and stream. PASS for the caller means
    LLMError was raised with a readable message (see `fault_verdict`)."""
    api_key, model, messages, params = apply_fault(fault, api_key, model, messages, params)
    (log or NullTrace()).event("fault", **fault)
    client = make_client(base_url, api_key, params.timeout_s, http_client=http_client)
    yield from stream_chat(client, model, messages, params, log=log, hook=hook)


def list_models(client: OpenAI) -> list[dict[str, Any]]:
    try:
        return [m.model_dump() for m in client.models.list().data]
    except Exception as exc:  # noqa: BLE001
        raise map_error(exc) from exc


__all__ = [
    "MAX_RETRIES",
    "MAX_TOOL_ROUNDS",
    "CancelHook",
    "ChatResult",
    "Delta",
    "EFFORTS",
    "GenerationParams",
    "LLMError",
    "Stopped",
    "Stopwatch",
    "apply_fault",
    "list_models",
    "make_client",
    "map_error",
    "open_with_retry",
    "retry_delay",
    "run_with_fault",
    "stream_chat",
]
