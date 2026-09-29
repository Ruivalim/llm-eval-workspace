import shutil
import subprocess

import pytest

from llm_eval.config import PROMPTS_DIR
from llm_eval.core.checks import CheckInput, run_check
from llm_eval.core.prompts import load_library


def v(t, params, answer, tool_calls=(), allow_code_exec=False):
    return run_check(
        t,
        params,
        CheckInput(answer=answer, tool_calls=list(tool_calls), allow_code_exec=allow_code_exec),
    )


def test_final_number_reads_last_line_only():
    assert v("final_number", {"expected": 4}, "3 daughters + 1 son\n\n4").ok
    assert not v("final_number", {"expected": 4}, "4\nso the answer is 3").ok
    assert not v("final_number", {"expected": 4}, "no digits here").ok


def test_final_number_ignores_thousands_separator():
    assert v("final_number", {"expected": 1152}, "Total: 1,152").ok
    assert v("final_number", {"expected": 1152}, "Total: 1.152").ok


def test_final_number_sees_through_a_trailing_code_fence():
    assert v("final_number", {"expected": 143}, "**Step 5:** 143\n\n```\n143\n```").ok
    assert v("final_number", {"expected": 143}, "```text\n143\n```").ok


def test_regex_multiline():
    p = {"pattern": r"(?im)^final:\s*(1|one)\b.*\bred\b"}
    assert v("regex", p, "blah\nFinal: 1 apple, in the red box").ok
    assert not v("regex", p, "Final: 2 apples, red box").ok


def test_contains_all_case_insensitive_by_default():
    assert v("contains_all", {"values": ["kill -9"]}, "run KILL -9 <pid>").ok
    assert not v("contains_all", {"values": ["kill -9"], "case_sensitive": True}, "KILL -9").ok


def test_exact_lines():
    assert v("exact_lines", {"count": 3, "prefix": "- "}, "- a\n- b\n- c\n").ok
    assert not v("exact_lines", {"count": 3, "prefix": "- "}, "intro\n- a\n- b\n- c").ok
    assert not v("exact_lines", {"count": 3, "prefix": "- "}, "* a\n* b\n* c").ok


def test_json_equals_accepts_fenced_and_rejects_wrong_types():
    exp = {"expected": {"service": "widget service", "port": 1234, "healthy": True}}
    assert v(
        "json_equals", exp, '```json\n{"service":"widget service","port":1234,"healthy":true}\n```'
    ).ok
    assert not v("json_equals", exp, '{"service":"widget service","port":"1234","healthy":true}').ok
    assert not v("json_equals", exp, "not json").ok


def test_html_complete():
    html = "<!DOCTYPE html><html><body><canvas></canvas>" + "x" * 200 + "</body></html>"
    assert v(
        "html_complete",
        {"min_chars": 100, "must_contain": ["<canvas"]},
        f"Sure!\n```html\n{html}\n```",
    ).ok
    assert not v("html_complete", {"min_chars": 10_000}, html).ok
    assert not v("html_complete", {"must_contain": ["<button"]}, html).ok
    assert not v("html_complete", {}, "<div>fragment</div>").ok


def test_tool_called_needs_every_tool_and_final_text():
    p = {"tools": ["get_weather", "calculate_math"], "final_contains": ["55212"]}
    assert v("tool_called", p, "It is 55212.", ["get_weather", "calculate_math"]).ok
    assert not v("tool_called", p, "It is 55212.", ["get_weather"]).ok
    # 55,212 is the same value as 55212: the needle is a number, and the prompt
    # asks for the value, not for a particular digit grouping. This assertion used
    # to demand a literal substring match, which failed four correct answers in the
    # weather-and-math battery.
    assert v("tool_called", p, "It is 55,212.", ["get_weather", "calculate_math"]).ok
    assert v("tool_called", p, "It is 55.212.", ["get_weather", "calculate_math"]).ok
    assert not v("tool_called", p, "It is 5,521.", ["get_weather", "calculate_math"]).ok
    assert not v("tool_called", p, "It is 55,21.", ["get_weather", "calculate_math"]).ok


def test_unknown_check_type_fails_loudly():
    r = v("does_not_exist", {}, "x")
    assert not r.ok and "unknown" in r.detail


@pytest.mark.skipif(not (shutil.which("bun") or shutil.which("node")), reason="needs a JS runtime")
def test_js_harness_runs_code():
    harness = 'if (total([1,2,3]) !== 6) throw new Error("bad");'
    good = "```js\nfunction total(items){ return items.reduce((a,b)=>a+b,0); }\n```"
    bad = "```js\nfunction total(items){ return 0; }\n```"
    assert v("js_harness", {"harness": harness}, good, allow_code_exec=True).ok
    assert not v("js_harness", {"harness": harness}, bad, allow_code_exec=True).ok
    assert not v(
        "js_harness",
        {"harness": harness, "must_not_contain": ["reduce"]},
        good,
        allow_code_exec=True,
    ).ok


def test_js_harness_off_unless_opted_in():
    """Model output is arbitrary code: the check refuses to run by default and says
    how to turn it on, instead of silently executing."""
    r = v("js_harness", {"harness": "x"}, "```js\nconst x = 5;\n```")
    assert r.infra and r.status == "ERROR" and "LLM_EVAL_ALLOW_CODE_EXEC" in r.detail


@pytest.mark.skipif(not (shutil.which("bun") or shutil.which("node")), reason="needs a JS runtime")
def test_js_harness_child_does_not_see_the_environment(monkeypatch, tmp_path):
    """The runner inherits no API key: env is PATH-only, and Bun must not auto-load
    the .env sitting in the working directory."""
    monkeypatch.setenv("AISERVER_API_KEY", "leak-me")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("AISERVER_API_KEY=leak-me-from-file\n")
    leak = (
        "if (process.env.AISERVER_API_KEY)"
        ' throw new Error("leaked " + process.env.AISERVER_API_KEY);'
    )
    r = v("js_harness", {"harness": leak}, "```js\nconst x = 1;\n```", allow_code_exec=True)
    assert r.ok, r.detail


def test_js_harness_failure_reports_the_error_not_the_source(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "/bin/node")
    stderr = (
        "file:///tmp/[eval1]:18\n"
        'if (x !== 6) throw new Error("bad value");\n'
        "\nError: bad value\n    at file:///tmp/[eval1]:18:1\n"
    )
    proc = subprocess.CompletedProcess([], 1, stdout="", stderr=stderr)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: proc)
    r = v("js_harness", {"harness": "x"}, "```js\nconst x = 5;\n```", allow_code_exec=True)
    assert not r.ok and "bad value" in r.detail and "const x" not in r.detail


def test_js_harness_without_runtime_is_an_infra_error(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    r = v("js_harness", {"harness": "x"}, "```js\nconst x = 5;\n```", allow_code_exec=True)
    assert r.infra and r.status == "ERROR", r


@pytest.mark.skipif(not (shutil.which("bun") or shutil.which("node")), reason="needs a JS runtime")
def test_battery_harness_prompts_pass_a_correct_implementation():
    """The shipped harnesses must accept a correct answer. One expectation was
    inconsistent (it merged [9,9] with [10,10] but kept [1,4] apart from [5,7])."""
    lib = load_library(PROMPTS_DIR)
    refs = {
        "merge-ranges": """
            function mergeRanges(ranges) {
              const sorted = ranges.map(([s, e]) => [s, e]).sort((a, b) => a[0] - b[0]);
              const merged = [];
              for (const [s, e] of sorted) {
                const last = merged[merged.length - 1];
                if (last && s <= last[1]) last[1] = Math.max(last[1], e);
                else merged.push([s, e]);
              }
              return merged;
            }
        """,
        "summarize-orders": """
            function summarizeOrders(orders) {
              const byCategory = {};
              for (const { category, quantity, price } of orders) {
                const row = (byCategory[category] ??= { category, units: 0, revenue: 0 });
                row.units += quantity;
                row.revenue += quantity * price;
              }
              return Object.values(byCategory).sort((a, b) => a.category.localeCompare(b.category));
            }
        """,
    }
    for pid, code in refs.items():
        p = lib.by_id(pid)
        r = v("js_harness", p.check.params, f"```js\n{code}\n```", allow_code_exec=True)
        assert r.ok, f"{pid}: {r.detail}"
