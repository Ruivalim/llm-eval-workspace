"""Stop means stop: cancel closes the model stream, the worker dies fast,
and no error row is saved. Uses FakeServer (buffered mock), so these pin the
wiring; the real unblock-a-parked-socket behavior is covered by the smoke run."""

import asyncio
import sqlite3
import threading
import time

import httpx
import pytest

from llm_eval import api
from llm_eval.config import Settings
from llm_eval.core import engine
from llm_eval.core.client import CancelHook, GenerationParams, Stopped, make_client, stream_chat
from llm_eval.core.engine import build_messages
from llm_eval.core.prompts import Library, Prompt
from llm_eval.core.trace import Trace
from llm_eval.db import crud
from tests.fake_server import FakeServer, sse
from tests.test_api import _use_fake
from tests.test_engine import _prompt

P = GenerationParams(timeout_s=5)
MSGS = [{"role": "user", "content": "hi"}]

LIB = Library(
    prompts=[
        Prompt.model_validate(
            {
                "id": "sum",
                "category": "logic",
                "eval_type": "text",
                "check": {"type": "final_number", "expected": 4},
                "locales": {"en": {"name": "Sum", "prompt": "2+2?"}},
            }
        ),
    ]
)


@pytest.fixture
def db(conn):
    sid = crud.add_server(conn, "s", "http://fake/v1", "k")
    server = crud.list_servers(conn)[0]
    run_id = crud.new_run(conn, sid, "m", "en")
    return conn, server, run_id


class _Stream:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


def test_hook_cancel_closes_registered_streams_and_is_idempotent():
    hook, a, b = CancelHook(), _Stream(), _Stream()
    hook.register(a)
    hook.register(b)
    assert not hook.cancelled
    hook.cancel()
    hook.cancel()
    assert hook.cancelled and (a.closed, b.closed) == (1, 1)


def test_hook_register_after_cancel_closes_immediately():
    hook = CancelHook()
    hook.cancel()
    s = _Stream()
    hook.register(s)
    assert s.closed == 1


def test_stream_chat_precancelled_raises_stopped_not_error():
    hook = CancelHook()
    hook.cancel()
    fake = FakeServer([sse([{"content": "4"}])])
    client = make_client("http://fake/v1", "k", P.timeout_s, http_client=fake.client())
    with pytest.raises(Stopped):
        list(stream_chat(client, "m", MSGS, P, hook=hook))


def test_stream_chat_stop_mid_create_still_stops():
    release = threading.Event()

    def gated(request):
        assert release.wait(timeout=5)
        return httpx.Response(
            200, content=sse([{"content": "4"}]), headers={"content-type": "text/event-stream"}
        )

    fake = FakeServer([gated])
    hook = CancelHook()
    client = make_client("http://fake/v1", "k", P.timeout_s, http_client=fake.client())
    outcome = {}

    def run():
        try:
            list(stream_chat(client, "m", MSGS, P, hook=hook))
        except Stopped:
            outcome["stopped"] = True

    t = threading.Thread(target=run, daemon=True)
    t.start()
    deadline = time.time() + 5
    while not fake.requests and time.time() < deadline:
        time.sleep(0.01)
    assert fake.requests, "worker never reached the model"
    hook.cancel()
    release.set()
    t.join(timeout=5)
    assert not t.is_alive() and outcome.get("stopped") is True


def test_engine_stream_stop_propagates_and_saves_nothing(db, tmp_path, monkeypatch):
    conn, server, _run_id = db
    monkeypatch.setattr(
        crud, "save_result", lambda *a, **k: (_ for _ in ()).throw(AssertionError("saved!"))
    )
    p, hook = _prompt(), CancelHook()
    hook.cancel()
    fake = FakeServer([sse([{"content": "4"}])])
    with Trace.start(tmp_path / "tr", p.id, "m") as tr, pytest.raises(Stopped):
        engine._stream(
            server, "m", p, build_messages(p, "en", None), P, None, fake.client(), tr, hook
        )


def test_evaluate_gen_close_saves_no_row_and_kills_worker(tmp_path, monkeypatch):
    """Closing the SSE stream (what a browser abort becomes inside uvicorn)
    must stop the worker and save nothing. Drives evaluate_events directly:
    the HTTP-disconnect-to-generator-close half belongs to the framework."""

    def slow_then_answer(request):
        time.sleep(2)
        return httpx.Response(
            200, content=sse([{"content": "4"}]), headers={"content-type": "text/event-stream"}
        )

    _use_fake(monkeypatch, FakeServer([slow_then_answer]))
    settings = Settings(
        tmp_path / "t.db",
        "http://fake/v1",
        "k",
        "en",
        tmp_path / "art",
        tmp_path / "tr",
        tmp_path / "logs",
    )
    s = api.State(settings, LIB)
    body = api.EvaluateIn(server_id=1, model="m", prompt_id="sum")

    async def drive():
        gen = api.evaluate_events(s, body)
        try:
            first = await gen.__anext__()
            assert "start" in first
        finally:
            await gen.aclose()

    asyncio.run(drive())
    # Worker was parked in create at close; past the 2s handler sleep it must
    # notice the cancel and die instead of saving.
    deadline = time.time() + 10
    while any(t.name == "eval-sum" for t in threading.enumerate()) and time.time() < deadline:
        time.sleep(0.05)
    assert not any(t.name == "eval-sum" for t in threading.enumerate())
    with sqlite3.connect(settings.db_path) as c:
        assert c.execute("select count(*) from results").fetchone()[0] == 0
