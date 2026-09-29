import io

import pytest

from llm_eval import cli
from llm_eval.core.checks import Verdict
from llm_eval.core.client import ChatResult, GenerationParams
from llm_eval.core.engine import Outcome
from llm_eval.core.prompts import Library, Prompt
from llm_eval.core.telemetry import Telemetry
from llm_eval.db import crud


def _p(pid, check_type="regex"):
    return Prompt.model_validate(
        {
            "id": pid,
            "category": "c",
            "eval_type": "text",
            "check": {"type": check_type, "pattern": "x"},
            "locales": {"en": {"name": pid, "prompt": "p"}},
        }
    )


def _outcome(status, ttft=100, tps=10.0):
    res = None if status == "ERROR" else ChatResult("x", "", [], Telemetry(ttft_ms=ttft, tps=tps))
    return Outcome(
        res, Verdict(status == "PASS", status.lower()), "err" if status == "ERROR" else None
    )


def test_mean_sd():
    assert cli.mean_sd([]) == "n/a"
    assert cli.mean_sd([5]) == "5"
    assert cli.mean_sd([100, 200]) == "150 ± 71"


def test_pick_server(conn):
    with pytest.raises(SystemExit, match="no servers"):
        cli.pick_server([], None)
    crud.add_server(conn, "a", "http://a/v1", None)
    crud.add_server(conn, "b", "http://b/v1", None)
    servers = crud.list_servers(conn)
    assert cli.pick_server(servers, None).name == "a"
    assert cli.pick_server(servers, "b").name == "b"
    with pytest.raises(SystemExit, match="unknown server"):
        cli.pick_server(servers, "zzz")


def test_run_battery_skips_manual_prompts_and_aggregates(conn):
    crud.add_server(conn, "a", "http://a/v1", None)
    server = crud.list_servers(conn)[0]
    lib = Library(prompts=[_p("auto1"), _p("manual", "none"), _p("auto2")])
    script = {
        "auto1": [_outcome("PASS", 100, 10), _outcome("PASS", 200, 20)],
        "auto2": [_outcome("FAIL"), _outcome("ERROR")],
    }
    calls = []

    def fake_eval(conn, run_id, server, model, p, locale, params, **kw):
        calls.append((p.id, model, locale, params.timeout_s))
        return script[p.id].pop(0)

    out = io.StringIO()
    s = cli.run_battery(
        conn, server, "m", lib, "en", 2, GenerationParams(timeout_s=9), fake_eval, out
    )
    assert [c[0] for c in calls] == ["auto1", "auto1", "auto2", "auto2"]
    assert calls[0][1:] == ("m", "en", 9)
    assert [x.status for x in s] == ["PASS", "FAIL"]
    text = out.getvalue()
    assert "PASS  auto1" in text and "150 ± 71" in text and "1/2 passed" in text
    assert not any(x.ok for x in s if x.prompt_id == "auto2")


def test_main_validates_args(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--model", "m", "--repeat", "0"])
    assert e.value.code == 2
    with pytest.raises(SystemExit):
        cli.main(["--model", " "])
    with pytest.raises(SystemExit):
        cli.main([])


def test_main_exit_code_follows_failures(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_EVAL_DB", str(tmp_path / "t.db"))
    monkeypatch.setenv("LLM_EVAL_LOGS", str(tmp_path / "logs"))
    # A fresh DB only gets a server when the environment asks for one.
    monkeypatch.setenv("LLM_EVAL_DEFAULT_BASE_URL", "http://fake/v1")
    lib = Library(prompts=[_p("a")])
    monkeypatch.setattr(cli, "load_library", lambda: lib)
    monkeypatch.setattr(cli.engine, "evaluate", lambda *a, **k: _outcome("PASS"))
    assert cli.main(["--model", "m"]) == 0
    monkeypatch.setattr(cli.engine, "evaluate", lambda *a, **k: _outcome("FAIL"))
    assert cli.main(["--model", "m"]) == 1


def test_effort_flag_is_validated():
    import pytest

    from llm_eval.cli import main

    with pytest.raises(SystemExit) as info:
        main(["--model", "m", "--effort", "turbo"])
    assert info.value.code == 2
