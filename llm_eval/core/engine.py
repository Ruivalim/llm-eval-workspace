"""One evaluation = one prompt against one model, end to end: build messages,
stream (with tool loop or fault injection), run the check, save the artifact and
the DB row. UI and CLI both go through `evaluate` so their numbers agree."""

from __future__ import annotations

import logging
import re
import sqlite3
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llm_eval.config import ARTIFACTS_DIR, TRACES_DIR
from llm_eval.core import anthropic_client, pricing, tool_mock
from llm_eval.core.checks import CheckInput, Verdict, run_check
from llm_eval.core.client import (
    CancelHook,
    ChatResult,
    Delta,
    GenerationParams,
    LLMError,
    Stopped,
    apply_fault,
    make_client,
    run_with_fault,
    stream_chat,
)
from llm_eval.core.html_extractor import extract_html
from llm_eval.core.prompts import Prompt
from llm_eval.core.telemetry import Telemetry
from llm_eval.core.trace import Trace
from llm_eval.db import crud
from llm_eval.db.crud import Server

OnDelta = Callable[[Delta], None]
log = logging.getLogger(__name__)


@dataclass
class Outcome:
    result: ChatResult | None
    verdict: Verdict
    error: str | None = None
    artifact_path: str = ""
    result_id: int | None = None
    trace_path: str = ""
    cost_usd: float | None = None
    cost_source: str = ""

    @property
    def status(self) -> str:
        if self.verdict.infra:
            return "ERROR"
        if self.error is not None and self.result is None and not self.verdict.ok:
            return "ERROR"
        return self.verdict.status


def build_messages(
    p: Prompt, locale: str, system_override: str | None, global_system: str = ""
) -> list[dict[str, Any]]:
    """System = global prefix (same for every prompt) + the prompt's own system
    text (or the per-run override), separated by a blank line."""
    txt = p.text(locale)
    own = txt.system_prompt if system_override is None else system_override
    system = "\n\n".join(part.strip() for part in (global_system, own) if part and part.strip())
    msgs: list[dict[str, Any]] = []
    if system.strip():
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": txt.prompt})
    return msgs


def is_auto(p: Prompt) -> bool:
    """Prompts the headless battery can judge alone."""
    return p.check.type != "none" or p.eval_type == "error_test"


def fault_verdict(error: str | None) -> Verdict:
    """error_test passes when the client surfaced one clean error line."""
    if error is None:
        return Verdict(False, "expected a failure, but the request succeeded")
    if "\n" in error.strip() or len(error) > 300:
        return Verdict(False, f"error surfaced but not clean: {error[:80]!r}")
    return Verdict(True, f"clean error: {error}")


def save_artifact(prompt_id: str, model: str, answer: str, directory: Path = ARTIFACTS_DIR) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    path = directory / f"{prompt_id}__{safe_model}__{stamp}.html"
    path.write_text(extract_html(answer), encoding="utf-8")
    return str(path)


def evaluate(
    conn: sqlite3.Connection,
    run_id: int,
    server: Server,
    model: str,
    p: Prompt,
    locale: str,
    params: GenerationParams,
    *,
    system_override: str | None = None,
    global_system: str | None = None,
    on_delta: OnDelta | None = None,
    http_client: Any = None,  # httpx.Client, or httpx2.Client for anthropic
    hook: CancelHook | None = None,
    artifacts_dir: Path | None = None,
    traces_dir: Path | None = None,
    allow_code_exec: bool = False,
) -> Outcome:
    artifacts_dir = artifacts_dir or ARTIFACTS_DIR
    if global_system is None:
        global_system = crud.get_setting(conn, crud.GLOBAL_SYSTEM_PROMPT)
    messages = build_messages(p, locale, system_override, global_system)
    params = replace(params, max_tokens=p.max_tokens)
    with Trace.start(traces_dir or TRACES_DIR, p.id, model) as tr:
        tr.event(
            "evaluate",
            run_id=run_id,
            prompt_id=p.id,
            eval_type=p.eval_type,
            locale=locale,
            server={
                "id": server.id,
                "name": server.name,
                "flavor": server.flavor,
                "base_url": server.base_url,
            },
            model=model,
            params=asdict(params),
            check=p.check.model_dump(),
            tools=p.tools,
            fault=p.fault,
            system_override=system_override is not None,
            global_system=global_system,
            messages=messages,
        )
        log.info("evaluate start prompt=%s model=%s run=%s trace=%s", p.id, model, run_id, tr.path)
        result, error = _stream(server, model, p, messages, params, on_delta, http_client, tr, hook)
        out = _finish(
            conn,
            run_id,
            server,
            model,
            p,
            params,
            result,
            error,
            artifacts_dir,
            tr,
            allow_code_exec,
        )
    return out


def _stream(
    server: Server,
    model: str,
    p: Prompt,
    messages: list[dict[str, Any]],
    params: GenerationParams,
    on_delta: OnDelta | None,
    http_client: Any,
    tr: Trace,
    hook: CancelHook | None = None,
) -> tuple[ChatResult | None, str | None]:
    result: ChatResult | None = None
    try:
        stream = _open_stream(server, model, p, messages, params, http_client, tr, hook)
        for item in stream:
            if isinstance(item, ChatResult):
                result = item
            elif on_delta is not None:
                on_delta(item)
    except Stopped:
        raise  # stop is not a result: no DB row, no error outcome
    except LLMError as exc:
        return None, str(exc)
    return result, None


def _open_stream(
    server: Server,
    model: str,
    p: Prompt,
    messages: list[dict[str, Any]],
    params: GenerationParams,
    http_client: Any,
    tr: Trace,
    hook: CancelHook | None,
) -> Iterator[Delta | ChatResult]:
    tools = tool_mock.schemas_for(p.tools) if p.eval_type == "tool_call" else None
    if server.flavor == "anthropic":
        api_key = server.api_key
        if p.eval_type == "error_test":
            api_key, model, messages, params = apply_fault(
                p.fault or {}, api_key, model, messages, params
            )
            tr.event("fault", **(p.fault or {}))
        client = anthropic_client.make_client(
            server.base_url, api_key, params.timeout_s, http_client
        )
        return anthropic_client.stream_messages(
            client, model, messages, params, tools=tools, log=tr, hook=hook
        )
    if p.eval_type == "error_test":
        return run_with_fault(
            p.fault or {},
            server.base_url,
            server.api_key,
            model,
            messages,
            params,
            http_client,
            log=tr,
            hook=hook,
        )
    client = make_client(server.base_url, server.api_key, params.timeout_s, http_client)
    return stream_chat(client, model, messages, params, tools=tools, log=tr, hook=hook)


def _finish(
    conn: sqlite3.Connection,
    run_id: int,
    server: Server,
    model: str,
    p: Prompt,
    params: GenerationParams,
    result: ChatResult | None,
    error: str | None,
    artifacts_dir: Path,
    tr: Trace,
    allow_code_exec: bool = False,
) -> Outcome:

    if p.eval_type == "error_test":
        verdict = fault_verdict(error)
    elif result is None:
        verdict = Verdict(False, error or "no result")
    else:
        verdict = run_check(
            p.check.type,
            p.check.params,
            CheckInput(result.answer, result.tool_calls, allow_code_exec),
        )
        if result.finish_reason == "length":
            cap = params.max_tokens or "server default"
            verdict = Verdict(
                verdict.ok, f"{verdict.detail} [truncated: finish_reason=length, max_tokens={cap}]"
            )
        elif result.finish_reason == "loop":
            kept = len(result.answer)
            verdict = Verdict(
                verdict.ok, f"{verdict.detail} [stopped: repetition loop, kept {kept} chars]"
            )
        elif result.finish_reason == "refusal":
            verdict = Verdict(verdict.ok, f"{verdict.detail} [stopped: the model refused]")

    artifact = ""
    if result is not None and p.eval_type == "render_html":
        artifact = save_artifact(p.id, model, result.answer, artifacts_dir)

    out = Outcome(result, verdict, error, artifact, trace_path=str(tr.path))
    if result is not None:
        out.cost_usd, out.cost_source = pricing.cost_of(conn, model, result.telemetry)
    out.result_id = crud.save_result(
        conn,
        run_id,
        p.id,
        p.eval_type,
        out.status,
        verdict.detail if error is None or p.eval_type == "error_test" else error,
        result.answer if result else "",
        result.telemetry if result else Telemetry(),
        reasoning=result.reasoning if result else "",
        artifact_path=artifact,
        trace_path=str(tr.path),
        # Anthropic runs never send sampling params; recording one would lie.
        temperature=None if server.flavor == "anthropic" else params.temperature,
        max_tokens=params.max_tokens,
        cost_usd=out.cost_usd,
        cost_source=out.cost_source,
    )
    tr.event(
        "outcome",
        result_id=out.result_id,
        status=out.status,
        detail=verdict.detail,
        error=error,
        artifact=artifact,
        telemetry=asdict(result.telemetry) if result else None,
        cost_usd=out.cost_usd,
        cost_source=out.cost_source,
    )
    log.info(
        "evaluate done prompt=%s model=%s result=%s status=%s ttft=%s tps=%s detail=%s",
        p.id,
        model,
        out.result_id,
        out.status,
        result.telemetry.ttft_ms if result else None,
        result.telemetry.tps if result else None,
        verdict.detail if error is None else error,
    )
    return out
