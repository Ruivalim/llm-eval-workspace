from pathlib import Path

import httpx
import pytest

from llm_eval.core.checks import Verdict
from llm_eval.core.client import GenerationParams
from llm_eval.core.engine import Outcome, build_messages, evaluate, fault_verdict, is_auto
from llm_eval.core.prompts import Prompt
from llm_eval.db import crud
from tests.fake_server import FakeServer, sse, tool_call_chunks

P = GenerationParams(timeout_s=5)


def _prompt(**over):
    base = {
        "id": "t",
        "category": "logic",
        "eval_type": "text",
        "check": {"type": "final_number", "expected": 4},
        "locales": {
            "en": {"name": "T", "system_prompt": "be terse", "prompt": "2+2?"},
            "pt-BR": {"name": "T", "system_prompt": "", "prompt": "2+2?"},
        },
    }
    base.update(over)
    return Prompt.model_validate(base)


@pytest.fixture(autouse=True)
def _traces_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr("llm_eval.core.engine.TRACES_DIR", tmp_path / "traces")


@pytest.fixture
def db(conn):
    sid = crud.add_server(conn, "s", "http://fake/v1", "k")
    server = crud.list_servers(conn)[0]
    run_id = crud.new_run(conn, sid, "m", "en")
    return conn, server, run_id


def test_build_messages_override_and_empty_system():
    p = _prompt()
    assert [m["role"] for m in build_messages(p, "en", None)] == ["system", "user"]
    assert build_messages(p, "en", "custom")[0]["content"] == "custom"
    assert [m["role"] for m in build_messages(p, "pt-BR", None)] == ["user"]
    assert [m["role"] for m in build_messages(p, "en", "")] == ["user"]


def test_build_messages_global_prefix():
    p = _prompt()
    assert (
        build_messages(p, "en", None, "Always answer in French.")[0]["content"]
        == "Always answer in French.\n\nbe terse"
    )
    assert build_messages(p, "pt-BR", None, "G")[0] == {"role": "system", "content": "G"}
    assert build_messages(p, "en", "", "  ")[0]["role"] == "user"
    assert build_messages(p, "en", "own", "G")[0]["content"] == "G\n\nown"


def test_evaluate_reads_global_system_from_db(db):
    conn, server, run_id = db
    crud.set_setting(conn, crud.GLOBAL_SYSTEM_PROMPT, "GLOBAL")
    srv = FakeServer([sse([{"content": "4"}])])
    evaluate(conn, run_id, server, "m", _prompt(), "en", P, http_client=srv.client())
    assert srv.body()["messages"][0]["content"] == "GLOBAL\n\nbe terse"


def test_is_auto():
    assert is_auto(_prompt())
    assert not is_auto(_prompt(check={"type": "none"}))
    assert is_auto(
        _prompt(eval_type="error_test", check={"type": "none"}, fault={"kind": "timeout"})
    )


def test_fault_verdict():
    assert not fault_verdict(None).ok
    assert fault_verdict("timeout: x").ok
    assert not fault_verdict("a\nb").ok
    assert not fault_verdict("x" * 400).ok


def test_infra_verdict_is_an_error_not_a_model_failure():
    """A missing JS runtime is the machine's problem, so it must not count
    against the model's pass rate."""
    out = Outcome(None, Verdict(False, "neither bun nor node on PATH", infra=True))
    assert out.status == "ERROR"
    assert Outcome(None, Verdict(False, "plain failure")).status == "FAIL"


def test_evaluate_text_pass_saves_row_and_omits_max_tokens_by_default(db):
    conn, server, run_id = db
    srv = FakeServer(
        [sse([{"content": "2+2 is\n4"}], {"prompt_tokens": 3, "completion_tokens": 5})]
    )
    seen = []
    out = evaluate(
        conn,
        run_id,
        server,
        "m",
        _prompt(),
        "en",
        P,
        on_delta=seen.append,
        http_client=srv.client(),
    )
    assert out.status == "PASS" and out.result_id
    assert [d.text for d in seen] == ["2+2 is\n4"]
    assert "max_tokens" not in srv.body()
    row = crud.history(conn)[0]
    assert row["status"] == "PASS" and row["tokens_out"] == 5 and row["answer"] == "2+2 is\n4"


def test_evaluate_fail_when_check_fails(db):
    conn, server, run_id = db
    srv = FakeServer([sse([{"content": "5"}])])
    out = evaluate(conn, run_id, server, "m", _prompt(), "en", P, http_client=srv.client())
    assert out.status == "FAIL" and "expected 4" in out.verdict.detail


def test_evaluate_error_status_when_server_down(db):
    conn, server, run_id = db

    def refuse(req):
        raise httpx.ConnectError("refused", request=req)

    out = evaluate(
        conn, run_id, server, "m", _prompt(), "en", P, http_client=FakeServer([refuse]).client()
    )
    assert out.status == "ERROR" and out.result is None and "connection failed" in out.error
    assert crud.history(conn)[0]["status"] == "ERROR"


def test_evaluate_error_test_passes_on_clean_error_and_fails_on_success(db):
    conn, server, run_id = db
    p = _prompt(
        eval_type="error_test", check={"type": "none"}, fault={"kind": "timeout", "seconds": 1}
    )

    def slow(req):
        raise httpx.ReadTimeout("slow", request=req)

    out = evaluate(conn, run_id, server, "m", p, "en", P, http_client=FakeServer([slow]).client())
    assert out.status == "PASS" and "timeout" in out.verdict.detail
    ok = FakeServer([sse([{"content": "fine"}])]).client()
    out = evaluate(conn, run_id, server, "m", p, "en", P, http_client=ok)
    assert out.status == "FAIL"


def test_evaluate_tool_call_passes_tools_and_records_calls(db):
    conn, server, run_id = db
    p = _prompt(
        eval_type="tool_call",
        tools=["calculate_math"],
        check={"type": "tool_called", "tools": ["calculate_math"], "final_contains": ["55212"]},
    )
    srv = FakeServer(
        [
            sse(tool_call_chunks("calculate_math", {"expression": "1284*43"})),
            sse([{"content": "55212"}]),
        ]
    )
    out = evaluate(conn, run_id, server, "m", p, "en", P, http_client=srv.client())
    assert out.status == "PASS" and out.result.tool_calls == ["calculate_math"]
    assert "tools" in srv.body(0)


def test_evaluate_render_html_writes_artifact(db, tmp_path):
    conn, server, run_id = db
    p = _prompt(eval_type="render_html", check={"type": "html_complete"})
    html = "<!DOCTYPE html><html><body>hi</body></html>"
    srv = FakeServer([sse([{"content": f"```html\n{html}\n```"}])])
    out = evaluate(
        conn,
        run_id,
        server,
        "my/model:1",
        p,
        "en",
        P,
        http_client=srv.client(),
        artifacts_dir=tmp_path,
    )
    assert out.status == "PASS"
    files = list(tmp_path.glob("t__my_model_1__*.html"))
    assert len(files) == 1 and files[0].read_text() == html
    assert crud.history(conn)[0]["artifact_path"] == str(files[0])


def test_evaluate_writes_a_full_trace(db, tmp_path):
    from llm_eval.core.trace import read_trace

    conn, server, run_id = db
    p = _prompt(
        eval_type="tool_call",
        tools=["calculate_math"],
        check={"type": "tool_called", "tools": ["calculate_math"]},
    )
    srv = FakeServer(
        [
            sse(tool_call_chunks("calculate_math", {"expression": "2*2"})),
            sse([{"content": "4"}], {"completion_tokens": 1}),
        ]
    )
    out = evaluate(
        conn,
        run_id,
        server,
        "my/model",
        p,
        "en",
        P,
        http_client=srv.client(),
        traces_dir=tmp_path / "tr",
    )
    path = Path(out.trace_path)
    assert path.is_file() and path.parent.parent == tmp_path / "tr"
    assert path.name.endswith("__t__my_model.jsonl")
    events = read_trace(path)
    kinds = [e["ev"] for e in events]
    assert kinds[0] == "evaluate" and kinds[-1] == "outcome"
    assert kinds.count("request") == 2 and "tool_call" in kinds and "tool_result" in kinds
    assert kinds.count("chunk") >= 4 and "round_end" in kinds and "result" in kinds
    first_req = next(e for e in events if e["ev"] == "request")
    assert (
        first_req["messages"][-1]["content"] == "2+2?"
        and first_req["tools"][0]["function"]["name"] == "calculate_math"
    )
    assert events[-1]["result_id"] == out.result_id and events[-1]["status"] == "PASS"
    assert all("t_ms" in e and "ts" in e for e in events)
    assert crud.history(conn)[0]["trace_path"] == str(path)


def test_evaluate_trace_records_errors(db, tmp_path):
    from llm_eval.core.trace import read_trace

    conn, server, run_id = db

    def refuse(req):
        raise httpx.ConnectError("refused", request=req)

    out = evaluate(
        conn,
        run_id,
        server,
        "m",
        _prompt(),
        "en",
        P,
        http_client=FakeServer([refuse]).client(),
        traces_dir=tmp_path,
    )
    kinds = [e["ev"] for e in read_trace(Path(out.trace_path))]
    assert kinds == ["evaluate", "request", "error", "outcome"]


def test_truncation_is_flagged_in_the_detail(db, tmp_path):
    conn, server, run_id = db
    srv = FakeServer([sse([{"content": "2+2 is\n4", "finish_reason": "length"}])])
    out = evaluate(
        conn, run_id, server, "m", _prompt(), "en", P, http_client=srv.client(), traces_dir=tmp_path
    )
    assert out.status == "PASS" and out.result.finish_reason == "length"
    assert "truncated: finish_reason=length, max_tokens=server default" in out.verdict.detail


def test_loop_cut_is_flagged_in_the_detail(db, tmp_path):
    conn, server, run_id = db
    block = "The answer is seven always. "
    srv = FakeServer([sse([{"content": block}] * 40)])
    out = evaluate(
        conn, run_id, server, "m", _prompt(), "en", P, http_client=srv.client(), traces_dir=tmp_path
    )
    assert out.result.finish_reason == "loop"
    assert "repetition loop" in out.verdict.detail


def test_prompt_max_tokens_is_sent_when_set(db):
    conn, server, run_id = db
    srv = FakeServer([sse([{"content": "4"}])])
    evaluate(conn, run_id, server, "m", _prompt(max_tokens=77), "en", P, http_client=srv.client())
    assert srv.body()["max_tokens"] == 77
    assert crud.history(conn)[0]["max_tokens"] == 77


def test_on_delta_raising_llm_error_stops_the_run_cleanly(db, tmp_path):
    from llm_eval.core.client import LLMError

    conn, server, run_id = db
    srv = FakeServer([sse([{"content": "a"}, {"content": "b"}, {"content": "c"}])])
    seen = []

    def stop_after_first(d):
        seen.append(d.text)
        raise LLMError("stopped by the client")

    out = evaluate(
        conn,
        run_id,
        server,
        "m",
        _prompt(),
        "en",
        P,
        on_delta=stop_after_first,
        http_client=srv.client(),
        traces_dir=tmp_path,
    )
    assert seen == ["a"]
    assert out.status == "ERROR" and out.error == "stopped by the client"
    assert crud.history(conn)[0]["detail"] == "stopped by the client"
