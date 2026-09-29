"""HTTP surface for the React UI. Everything the UI does goes through here;
the CLI shares `core/engine.py` with it. Run with `make api`.

Streaming: POST /api/evaluate answers with Server-Sent Events. `delta` events
carry the live text, one final `outcome` event carries the saved result."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
from collections.abc import AsyncIterator
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.gzip import GZipMiddleware

from llm_eval.config import (
    DATA_DIR,
    ROOT,
    SUPPORTED_LOCALES,
    Settings,
    load_settings,
)
from llm_eval.core import anthropic_client, engine, pricing, unsloth
from llm_eval.core.client import (
    EFFORTS,
    CancelHook,
    Delta,
    GenerationParams,
    Stopped,
    list_models,
    make_client,
    map_error,
)
from llm_eval.core.prompts import Library, load_library
from llm_eval.core.trace import read_trace
from llm_eval.db import crud
from llm_eval.db.connection import connect
from llm_eval.db.init_db import init
from llm_eval.logging_setup import configure as configure_logging

log = logging.getLogger(__name__)

WEB_DIST = ROOT / "web" / "dist"
# Sandbox sem `allow-same-origin`: scripts do modelo rodam, mas em origem opaca.
ARTIFACT_CSP = "sandbox allow-scripts"


class State:
    def __init__(self, settings: Settings, library: Library | None = None):
        self.settings = settings
        configure_logging(settings.logs_dir)
        init(settings).close()
        self._conn = threading.local()
        self.library = library or load_library()
        self.categories = yaml.safe_load((DATA_DIR / "categories.yaml").read_text(encoding="utf-8"))

    @property
    def conn(self) -> sqlite3.Connection:
        """One SQLite connection per request or evaluation worker thread."""
        conn = getattr(self._conn, "conn", None)
        if conn is None:
            conn = connect(self.settings.db_path)
            self._conn.conn = conn
        return conn


class StaticGzip(GZipMiddleware):
    """Gzip for the built UI (the JS bundle is ~340 kB, and Lighthouse measures the
    page under throttling). `/api` is left alone: compressing a Server-Sent Events
    stream trades first-chunk latency for bytes nobody asked to save."""

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"].startswith("/api"):
            await self.app(scope, receive, send)
            return
        await super().__call__(scope, receive, send)


def spa_target(root: Path, path: str) -> Path | None:
    """The file inside the built UI that `path` points at, or None when it is empty
    or escapes `root`. Starlette URL-decodes the path before the route runs, so
    `..%2f` and `../` arrive here the same way: resolve first and only serve what
    is still inside the dist directory, or the catch-all becomes a file server."""
    if not path:
        return None
    resolved_root = root.resolve()
    target = (resolved_root / path).resolve()
    if target.is_relative_to(resolved_root) and target.is_file():
        return target
    return None


def create_app(settings: Settings | None = None, library: Library | None = None) -> FastAPI:
    app = FastAPI(title="llm-eval-workspace", docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.add_middleware(StaticGzip, minimum_size=500)
    app.state.state = State(settings or load_settings(), library)
    app.include_router(router)
    if WEB_DIST.exists():
        app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            if path == "api" or path.startswith("api/"):
                raise HTTPException(404, "no such API route")
            target = spa_target(WEB_DIST, path)
            return FileResponse(target or WEB_DIST / "index.html")

    return app


def get_state(request: Request) -> State:
    return request.app.state.state


from fastapi import APIRouter  # noqa: E402

StateDep = Annotated[State, Depends(get_state)]
router = APIRouter(prefix="/api")

# ---- catalogue --------------------------------------------------------------


@router.get("/meta")
def meta(s: StateDep) -> dict[str, Any]:
    return {
        "locales": list(SUPPORTED_LOCALES),
        "default_locale": s.settings.locale,
        "categories": s.categories,
        "prompts": [{**p.model_dump(), "auto": engine.is_auto(p)} for p in s.library.prompts],
    }


# ---- servers ----------------------------------------------------------------


class ServerIn(BaseModel):
    name: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    api_key: str | None = None
    flavor: str = "openai"


def _server(s: State, server_id: int) -> crud.Server:
    for srv in crud.list_servers(s.conn):
        if srv.id == server_id:
            return srv
    raise HTTPException(404, f"server {server_id} not found")


def _public(srv: crud.Server) -> dict[str, Any]:
    d = asdict(srv)
    d["has_key"] = bool(srv.api_key)
    del d["api_key"]
    return d


@router.get("/servers")
def servers(s: StateDep) -> list[dict[str, Any]]:
    return [_public(x) for x in crud.list_servers(s.conn)]


@router.post("/servers", status_code=201)
def add_server(body: ServerIn, s: StateDep) -> dict[str, Any]:
    if body.flavor not in crud.FLAVORS:
        raise HTTPException(422, f"flavor must be one of {', '.join(crud.FLAVORS)}")
    try:
        sid = crud.add_server(s.conn, body.name, body.base_url, body.api_key or None, body.flavor)
    except sqlite3.IntegrityError as exc:
        raise HTTPException(409, f"a server named {body.name!r} already exists") from exc
    return _public(_server(s, sid))


class ServerUpdate(ServerIn):
    api_key: str | None = None


@router.put("/servers/{server_id}")
def update_server(server_id: int, body: ServerUpdate, s: StateDep) -> dict[str, Any]:
    current = _server(s, server_id)
    if body.flavor not in crud.FLAVORS:
        raise HTTPException(422, f"flavor must be one of {', '.join(crud.FLAVORS)}")
    api_key = body.api_key if "api_key" in body.model_fields_set else current.api_key
    try:
        crud.update_server(s.conn, server_id, body.name, body.base_url, api_key, body.flavor)
    except sqlite3.IntegrityError as exc:
        raise HTTPException(409, f"a server named {body.name!r} already exists") from exc
    return _public(_server(s, server_id))


@router.delete("/servers/{server_id}", status_code=204)
def delete_server(server_id: int, s: StateDep) -> None:
    _server(s, server_id)
    try:
        crud.delete_server(s.conn, server_id)
    except sqlite3.IntegrityError as exc:
        raise HTTPException(409, "this server has runs in the history") from exc


@router.get("/servers/{server_id}/models")
def models(server_id: int, s: StateDep) -> list[dict[str, Any]]:
    srv = _server(s, server_id)
    try:
        if srv.flavor == "unsloth":
            raw = unsloth.models(srv.base_url, srv.api_key, timeout_s=20)
        elif srv.flavor == "anthropic":
            raw = anthropic_client.list_models(
                anthropic_client.make_client(srv.base_url, srv.api_key, 20)
            )
        else:
            raw = list_models(make_client(srv.base_url, srv.api_key, 20))
    except Exception as exc:  # noqa: BLE001 - one readable line for the UI
        raise HTTPException(502, str(map_error(exc))) from exc
    return [{"id": m["id"], "loaded": bool(m.get("loaded"))} for m in raw]


class LoadIn(BaseModel):
    model: str


@router.post("/servers/{server_id}/load")
def load_model(server_id: int, body: LoadIn, s: StateDep) -> dict[str, str]:
    srv = _server(s, server_id)
    if srv.flavor != "unsloth":
        raise HTTPException(400, "only unsloth servers can load models")
    try:
        unsloth.load(srv.base_url, srv.api_key, body.model, wait_s=600)
    except TimeoutError as exc:
        raise HTTPException(504, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, str(map_error(exc))) from exc
    return {"loaded": body.model}


# ---- settings -----------------------------------------------------------------


class SettingsOut(BaseModel):
    global_system_prompt: str = ""


@router.get("/settings")
def get_settings(s: StateDep) -> SettingsOut:
    return SettingsOut(global_system_prompt=crud.get_setting(s.conn, crud.GLOBAL_SYSTEM_PROMPT))


@router.put("/settings")
def put_settings(body: SettingsOut, s: StateDep) -> SettingsOut:
    crud.set_setting(s.conn, crud.GLOBAL_SYSTEM_PROMPT, body.global_system_prompt)
    log.info("global system prompt updated (%d chars)", len(body.global_system_prompt))
    return get_settings(s)


# ---- prices -------------------------------------------------------------------


class PriceIn(BaseModel):
    model: str = Field(min_length=1)
    input_per_mtok: float = Field(ge=0, le=10_000)
    output_per_mtok: float = Field(ge=0, le=10_000)


@router.get("/prices")
def prices(s: StateDep) -> dict[str, Any]:
    """Your prices, and the built-in list prices they override."""
    return {
        "prices": [dict(r) for r in crud.list_prices(s.conn)],
        "builtin": [
            {"model": m, "input_per_mtok": i, "output_per_mtok": o}
            for m, (i, o) in pricing.BUILTIN.items()
        ],
        "builtin_as_of": pricing.BUILTIN_AS_OF,
    }


@router.put("/prices")
def put_price(body: PriceIn, s: StateDep) -> dict[str, Any]:
    model = body.model.strip()
    filled = crud.set_price(s.conn, model, body.input_per_mtok, body.output_per_mtok)
    log.info(
        "price set model=%s in=%s out=%s filled=%d",
        model,
        body.input_per_mtok,
        body.output_per_mtok,
        filled,
    )
    return {"model": model, "filled": filled}


@router.delete("/prices", status_code=204)
def delete_price(model: str, s: StateDep) -> None:
    if not crud.delete_price(s.conn, model):
        raise HTTPException(404, f"no price set for {model!r}")


# ---- evaluate (SSE) -----------------------------------------------------------


class EvaluateIn(BaseModel):
    server_id: int
    model: str = Field(min_length=1)
    prompt_id: str
    locale: str = "en"
    temperature: float = 0.0
    top_p: float = 1.0
    timeout_s: float = 600.0
    effort: Literal[EFFORTS] = "default"  # type: ignore[valid-type]
    system_override: str | None = None
    # How many of these requests the batch has in flight. Recorded, not enforced:
    # the UI is what runs them at once; this only labels the run.
    parallel: int = Field(default=1, ge=1, le=16)


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _outcome_json(out: engine.Outcome) -> dict[str, Any]:
    r = out.result
    return {
        "result_id": out.result_id,
        "status": out.status,
        "detail": out.verdict.detail,
        "error": out.error,
        "artifact": Path(out.artifact_path).name if out.artifact_path else None,
        "trace": bool(out.trace_path),
        "finish_reason": r.finish_reason if r else None,
        "answer": r.answer if r else "",
        "reasoning": r.reasoning if r else "",
        "tool_calls": r.tool_calls if r else [],
        "telemetry": asdict(r.telemetry) if r else None,
        "cost_usd": out.cost_usd,
        "cost_source": out.cost_source,
    }


def evaluate_events(s: State, body: EvaluateIn) -> AsyncIterator[str]:
    srv = _server(s, body.server_id)
    try:
        p = s.library.by_id(body.prompt_id)
    except KeyError as exc:
        raise HTTPException(404, f"prompt {body.prompt_id!r} not found") from exc
    if body.locale not in SUPPORTED_LOCALES:
        raise HTTPException(422, f"locale must be one of {SUPPORTED_LOCALES}")
    params = GenerationParams(
        body.temperature, body.top_p, p.max_tokens, body.timeout_s, effort=body.effort
    )
    loop = asyncio.get_running_loop()
    q: asyncio.Queue[Delta | engine.Outcome | Exception] = asyncio.Queue()
    hook = CancelHook()

    def push(item: Delta | engine.Outcome | Exception) -> None:
        loop.call_soon_threadsafe(q.put_nowait, item)

    def on_delta(d: Delta) -> None:
        if hook.cancelled:
            raise Stopped("stopped by the client")
        push(d)

    def work() -> None:
        try:
            run_id = crud.new_run(
                s.conn, srv.id, body.model, body.locale, parallel=body.parallel, effort=body.effort
            )
            out = engine.evaluate(
                s.conn,
                run_id,
                srv,
                body.model,
                p,
                body.locale,
                params,
                system_override=body.system_override,
                on_delta=on_delta,
                hook=hook,
                artifacts_dir=s.settings.artifacts_dir,
                traces_dir=s.settings.traces_dir,
                allow_code_exec=s.settings.allow_code_exec,
            )
            push(out)
        except Stopped:
            pass  # nobody listens anymore; Stopped is not a result, save nothing
        except Exception as exc:  # noqa: BLE001 - forwarded to the stream
            log.exception("evaluate crashed prompt=%s model=%s", p.id, body.model)
            push(exc)

    async def gen() -> AsyncIterator[str]:
        threading.Thread(target=work, daemon=True, name=f"eval-{p.id}").start()
        try:
            yield _sse("start", {"prompt_id": p.id, "model": body.model, "eval_type": p.eval_type})
            while True:
                item = await q.get()
                if isinstance(item, Delta):
                    yield _sse("delta", {"kind": item.kind, "text": item.text})
                elif isinstance(item, engine.Outcome):
                    yield _sse("outcome", _outcome_json(item))
                    return
                else:
                    yield _sse("error", {"message": str(item)})
                    return
        finally:
            # Client disconnect, reload or shutdown: unblock the worker and close
            # the model stream, so the server drops the run instead of finishing
            # it unheard.
            hook.cancel()

    return gen()


@router.post("/evaluate")
async def evaluate(body: EvaluateIn, s: StateDep) -> StreamingResponse:
    return StreamingResponse(
        evaluate_events(s, body),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---- results ------------------------------------------------------------------


@router.get("/history")
def history(
    s: StateDep,
    model: str | None = None,
    prompt_id: str | None = None,
    status: str | None = None,
    locale: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    rows = crud.history_filtered(
        s.conn,
        model=model,
        prompt_id=prompt_id,
        status=status,
        locale=locale,
        limit=min(limit, 2000),
    )
    return [_row(r) for r in rows]


@router.get("/leaderboard")
def leaderboard(
    s: StateDep, model: str | None = None, prompt_id: str | None = None, locale: str | None = None
) -> list[dict[str, Any]]:
    return [
        dict(r) for r in crud.leaderboard(s.conn, model=model, prompt_id=prompt_id, locale=locale)
    ]


@router.get("/matrix")
def matrix(
    s: StateDep, model: str | None = None, prompt_id: str | None = None, locale: str | None = None
) -> list[dict[str, Any]]:
    return [dict(r) for r in crud.matrix(s.conn, model=model, prompt_id=prompt_id, locale=locale)]


@router.get("/results/{result_id}")
def result(result_id: int, s: StateDep) -> dict[str, Any]:
    row = crud.get_result(s.conn, result_id)
    if row is None:
        raise HTTPException(404, "result not found")
    return _row(row)


def _drop_files(s: State, rows: list[sqlite3.Row]) -> None:
    """Remove the artifact and trace files of the deleted results. Paths come from
    the DB, so only unlink what resolves inside the configured directories."""
    roots = (s.settings.artifacts_dir.resolve(), s.settings.traces_dir.resolve())
    for row in rows:
        # sqlite3.Row has no .get, so copy it: the columns differ per caller.
        fields = dict(row)
        for key in ("artifact_path", "trace_path"):
            raw = fields.get(key, "")
            if not raw:
                continue
            path = Path(raw)
            if any(path.resolve().is_relative_to(root) for root in roots):
                path.unlink(missing_ok=True)


@router.delete("/results/{result_id}", status_code=204)
def delete_result(result_id: int, s: StateDep) -> None:
    row = crud.delete_result(s.conn, result_id)
    if row is None:
        raise HTTPException(404, "result not found")
    _drop_files(s, [row])


@router.delete("/results")
def clear_results(
    s: StateDep,
    model: str | None = None,
    prompt_id: str | None = None,
    status: str | None = None,
    locale: str | None = None,
) -> dict[str, Any]:
    """Delete saved results, all or filtered like /history. Prompts, servers and
    settings are untouched."""
    count, rows = crud.clear_results(
        s.conn, model=model, prompt_id=prompt_id, status=status, locale=locale
    )
    _drop_files(s, rows)
    return {"deleted": count}


class ScoreIn(BaseModel):
    score: int = Field(ge=0, le=10)
    comment: str = ""


@router.post("/results/{result_id}/score")
def score(result_id: int, body: ScoreIn, s: StateDep) -> dict[str, Any]:
    if crud.get_result(s.conn, result_id) is None:
        raise HTTPException(404, "result not found")
    crud.set_human_score(s.conn, result_id, body.score, body.comment)
    return {"result_id": result_id, "score": body.score, "comment": body.comment}


@router.get("/results/{result_id}/trace")
def trace(result_id: int, s: StateDep, raw: bool = False) -> Any:
    """The execution trace of one result: JSON array of events, or the raw
    JSONL file with ?raw=1."""
    row = crud.get_result(s.conn, result_id)
    if row is None:
        raise HTTPException(404, "result not found")
    path = Path(row["trace_path"]) if row["trace_path"] else None
    if path is None or not path.is_file():
        raise HTTPException(404, "no trace file for this result")
    if raw:
        return FileResponse(path, media_type="application/x-ndjson", filename=path.name)
    return read_trace(path)


@router.get("/artifacts/{name}")
def artifact(name: str, s: StateDep) -> FileResponse:
    if "/" in name or ".." in name or not name.endswith(".html"):
        raise HTTPException(400, "bad artifact name")
    path = s.settings.artifacts_dir / name
    if not path.is_file():
        raise HTTPException(404, "artifact not found")
    # Model-generated HTML, opened in a new tab by the UI. CSP `sandbox` applies to
    # a top-level document, so the tab lands in an opaque origin just like the inline
    # iframe and cannot reach the API (which has no auth).
    return FileResponse(
        path, media_type="text/html", headers={"Content-Security-Policy": ARTIFACT_CSP}
    )


def _row(r: sqlite3.Row) -> dict[str, Any]:
    d = dict(r)
    d["artifact"] = Path(d["artifact_path"]).name if d.get("artifact_path") else None
    d.pop("artifact_path", None)
    d["trace"] = bool(d.pop("trace_path", ""))
    return d


def main() -> None:
    import os

    import uvicorn

    # Loopback by default, same reason as the Makefile. A container sets
    # LLM_EVAL_HOST=0.0.0.0 (the Dockerfile already passes it to uvicorn).
    uvicorn.run(
        create_app(),
        host=os.getenv("LLM_EVAL_HOST", "127.0.0.1"),
        port=int(os.getenv("LLM_EVAL_PORT", "8600")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
