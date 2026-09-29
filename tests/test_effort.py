"""Thinking effort: what each flavor sends, that it survives the engine, and
that results at different efforts stay apart."""

import pytest

from llm_eval.core.anthropic_client import make_client as anthropic_client
from llm_eval.core.anthropic_client import stream_messages
from llm_eval.core.client import (
    EFFORTS,
    GenerationParams,
    apply_fault,
    make_client,
    stream_chat,
)
from llm_eval.core.telemetry import Telemetry
from llm_eval.db import crud
from tests.fake_anthropic import FakeAnthropic, messages_sse, text_block
from tests.fake_server import FakeServer, sse

MSGS = [{"role": "user", "content": "hi"}]


def _openai_body(effort):
    srv = FakeServer([sse([{"content": "ok"}])])
    client = make_client("http://fake/v1", "k", 5, http_client=srv.client())
    list(stream_chat(client, "m", MSGS, GenerationParams(timeout_s=5, effort=effort)))
    return srv.body()


def _anthropic_body(effort):
    srv = FakeAnthropic([messages_sse([text_block("ok")])])
    client = anthropic_client("https://api.anthropic.com", "k", 5, srv.client())
    params = GenerationParams(max_tokens=100, timeout_s=5, effort=effort)
    list(stream_messages(client, "m", MSGS, params))
    return srv.body()


def test_default_sends_nothing_anywhere():
    body = _openai_body("default")
    assert "reasoning_effort" not in body and "thinking" not in body
    body = _anthropic_body("default")
    assert "output_config" not in body and "thinking" not in body


@pytest.mark.parametrize("level", ["low", "medium", "high", "xhigh", "max"])
def test_levels(level):
    body = _openai_body(level)
    assert body["reasoning_effort"] == level and "thinking" not in body
    body = _anthropic_body(level)
    assert body["output_config"] == {"effort": level} and "thinking" not in body


def test_off_disables_thinking_and_sends_no_level():
    body = _openai_body("off")
    assert body["thinking"] == {"type": "disabled"} and "reasoning_effort" not in body
    body = _anthropic_body("off")
    assert body["thinking"] == {"type": "disabled"} and "output_config" not in body


def test_fault_injection_keeps_the_effort():
    params = GenerationParams(timeout_s=600, effort="high")
    _, _, _, broken = apply_fault({"kind": "timeout", "seconds": 1}, "k", "m", MSGS, params)
    assert broken.timeout_s == 1 and broken.effort == "high"


def test_engine_keeps_the_effort_when_it_sets_max_tokens(conn, tmp_path):
    from llm_eval.core.engine import evaluate
    from llm_eval.core.prompts import Prompt

    sid = crud.add_server(conn, "s", "http://fake/v1", "k")
    server = crud.list_servers(conn)[0]
    run = crud.new_run(conn, sid, "m", "en", effort="low")
    p = Prompt.model_validate(
        {
            "id": "t",
            "category": "logic",
            "eval_type": "text",
            "max_tokens": 50,
            "check": {"type": "final_number", "expected": 4},
            "locales": {"en": {"name": "T", "prompt": "2+2?"}},
        }
    )
    srv = FakeServer([sse([{"content": "4"}])])
    evaluate(
        conn,
        run,
        server,
        "m",
        p,
        "en",
        GenerationParams(timeout_s=5, effort="low"),
        http_client=srv.client(),
        traces_dir=tmp_path,
    )
    assert srv.body()["reasoning_effort"] == "low" and srv.body()["max_tokens"] == 50


def _result(conn, model, effort, prompt, status):
    sid = (
        crud.list_servers(conn)[0].id
        if crud.list_servers(conn)
        else crud.add_server(conn, "s", "http://x/v1", None)
    )
    run = crud.new_run(conn, sid, model, "en", effort=effort)
    crud.save_result(conn, run, prompt, "text", status, "", "", Telemetry())


def test_leaderboard_and_matrix_keep_efforts_apart(conn):
    _result(conn, "m", "high", "p1", "PASS")
    _result(conn, "m", "high", "p2", "PASS")
    _result(conn, "m", "off", "p1", "FAIL")
    _result(conn, "m", "off", "p2", "PASS")
    rows = {r["effort"]: r for r in crud.leaderboard(conn)}
    assert set(rows) == {"high", "off"}
    assert rows["high"]["pass_rate"] == 100.0 and rows["off"]["pass_rate"] == 50.0
    assert rows["high"]["runs"] == 2 and rows["off"]["runs"] == 2
    # the latest run per prompt is picked within one effort, not across them
    cells = {(c["effort"], c["prompt_id"]): c["status"] for c in crud.matrix(conn)}
    assert cells == {
        ("high", "p1"): "PASS",
        ("high", "p2"): "PASS",
        ("off", "p1"): "FAIL",
        ("off", "p2"): "PASS",
    }
    assert {r["effort"] for r in crud.history_filtered(conn)} == {"high", "off"}


def test_old_runs_read_as_default(tmp_path):
    import sqlite3

    from llm_eval.db.connection import connect
    from llm_eval.db.schema import SCHEMA

    old = tmp_path / "old.db"
    raw = sqlite3.connect(old)
    stripped = "".join(
        ln for ln in SCHEMA.splitlines(True) if not ln.lstrip().startswith("effort ")
    )
    assert stripped != SCHEMA
    raw.executescript(stripped)
    raw.execute("INSERT INTO servers (name, base_url) VALUES ('s', 'http://x/v1')")
    raw.execute("INSERT INTO runs (server_id, model) VALUES (1, 'm')")
    raw.commit()
    raw.close()
    c = connect(old)
    assert c.execute("SELECT effort FROM runs").fetchone()[0] == "default"
    c.close()


def test_efforts_list_is_what_the_ui_offers():
    assert EFFORTS == ("default", "off", "low", "medium", "high", "xhigh", "max")
