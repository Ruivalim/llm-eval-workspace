"""Automatic pass/fail checks, selected by `check.type` in the prompt YAML.
Each checker gets the final answer (and, for tool calls, the trace) and returns
a Verdict. Add a checker = add a function and register it in CHECKERS."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_eval.core.html_extractor import extract_code, extract_html


@dataclass(frozen=True)
class Verdict:
    ok: bool
    detail: str
    infra: bool = False  # the check could not run: environment, not the model's fault

    @property
    def status(self) -> str:
        if self.infra:
            return "ERROR"
        return "PASS" if self.ok else "FAIL"


@dataclass
class CheckInput:
    answer: str
    tool_calls: list[str] = field(default_factory=list)
    allow_code_exec: bool = False


Checker = Callable[[CheckInput, dict[str, Any]], Verdict]


# 55,212 and 55.212 are the same number as 55212; a check on a value should not
# fail on the typography of digit grouping.
_THOUSANDS = re.compile(r"(?<=\d)[.,](?=\d{3}\b)")
# A lone fence line is markup, not the model's answer, so it never counts as the
# line that carries the final number.
_FENCE = re.compile(r"^```[\w.+-]*$")


def _strip_thousands(text: str) -> str:
    return _THOUSANDS.sub("", text)


def _content_lines(text: str) -> list[str]:
    return [
        ln.strip()
        for ln in text.strip().splitlines()
        if ln.strip() and not _FENCE.match(ln.strip())
    ]


def _last_number(text: str) -> str | None:
    lines = _content_lines(text)
    if not lines:
        return None
    last = _strip_thousands(lines[-1])
    nums = re.findall(r"-?\d+(?:[.,]\d+)?", last)
    return nums[-1].replace(",", ".") if nums else None


def check_none(inp: CheckInput, p: dict[str, Any]) -> Verdict:
    return Verdict(True, "no automatic check")


def check_final_number(inp: CheckInput, p: dict[str, Any]) -> Verdict:
    expected = str(p["expected"])
    got = _last_number(inp.answer)
    if got is None:
        return Verdict(False, "no number on the last line")
    ok = float(got) == float(expected)
    return Verdict(ok, f"last number {got}, expected {expected}")


def check_regex(inp: CheckInput, p: dict[str, Any]) -> Verdict:
    pattern = re.compile(p["pattern"])
    ok = pattern.search(inp.answer) is not None
    return Verdict(ok, f"regex {'matched' if ok else 'did not match'}: {p['pattern']}")


def check_contains_all(inp: CheckInput, p: dict[str, Any]) -> Verdict:
    hay = inp.answer if p.get("case_sensitive") else inp.answer.lower()
    missing = [v for v in p["values"] if (v if p.get("case_sensitive") else v.lower()) not in hay]
    if missing:
        return Verdict(False, f"missing: {', '.join(missing)}")
    return Verdict(True, f"all present: {', '.join(p['values'])}")


def check_exact_lines(inp: CheckInput, p: dict[str, Any]) -> Verdict:
    lines = [ln.strip() for ln in inp.answer.strip().splitlines() if ln.strip()]
    want = int(p["count"])
    if len(lines) != want:
        return Verdict(False, f"expected {want} lines, got {len(lines)}")
    prefix = p.get("prefix")
    if prefix and not all(ln.startswith(prefix) for ln in lines):
        return Verdict(False, f"not every line starts with {prefix!r}")
    return Verdict(True, f"exactly {want} lines")


def check_json_equals(inp: CheckInput, p: dict[str, Any]) -> Verdict:
    try:
        obj = json.loads(extract_code(inp.answer))
    except json.JSONDecodeError:
        return Verdict(False, "did not parse as JSON")
    for key, val in p["expected"].items():
        if obj.get(key) != val:
            return Verdict(False, f"{key}: got {obj.get(key)!r}, expected {val!r}")
    return Verdict(True, "JSON matches")


def check_html_complete(inp: CheckInput, p: dict[str, Any]) -> Verdict:
    html = extract_html(inp.answer)
    low = html.lower()
    if "<html" not in low or "</html>" not in low:
        return Verdict(False, "no complete <html>...</html> document")
    min_chars = int(p.get("min_chars", 0))
    if len(html) < min_chars:
        return Verdict(False, f"HTML too short: {len(html)} < {min_chars} chars")
    missing = [t for t in p.get("must_contain", []) if t.lower() not in low]
    if missing:
        return Verdict(False, f"HTML lacks: {', '.join(missing)}")
    return Verdict(True, f"HTML document, {len(html)} chars; needs human review")


def check_tool_called(inp: CheckInput, p: dict[str, Any]) -> Verdict:
    missing = [t for t in p["tools"] if t not in inp.tool_calls]
    if missing:
        return Verdict(False, f"tools not called: {', '.join(missing)}")
    hay = _strip_thousands(inp.answer)
    for needle in p.get("final_contains", []):
        if _strip_thousands(needle) not in hay:
            return Verdict(False, f"final answer lacks {needle!r}")
    return Verdict(True, f"called {', '.join(p['tools'])}")


def _harness_error(stderr: str) -> str:
    """The interesting line of a Node/Bun stack trace is the message, not the
    dumped source around it."""
    first = next((ln for ln in stderr.splitlines() if "Error" in ln), None)
    return " ".join((first or stderr).split())[:200]


def check_js_harness(inp: CheckInput, p: dict[str, Any]) -> Verdict:
    """Run extracted JS plus a harness that throws on failure. Needs node or bun.

    This executes model-generated code, so it stays off unless the operator opts in
    with LLM_EVAL_ALLOW_CODE_EXEC=1. The child gets a minimal environment (PATH
    only) and Bun is told not to auto-load .env, or a `process.env` line in the
    answer reads the API key straight out of the repo. There is still no sandbox
    around the filesystem or the network: see SECURITY.md."""
    if not inp.allow_code_exec:
        return Verdict(
            False,
            "code execution is disabled; set LLM_EVAL_ALLOW_CODE_EXEC=1 to run this check",
            infra=True,
        )
    runner = shutil.which("bun") or shutil.which("node")
    if not runner:
        return Verdict(False, "neither bun nor node on PATH", infra=True)
    code = extract_code(inp.answer)
    for bad in p.get("must_not_contain", []):
        if bad in code:
            return Verdict(False, f"code still contains {bad!r}")
    for pat in p.get("must_match", []):
        if not re.search(pat, code):
            return Verdict(False, f"code does not match {pat!r}")
    src = f"{code}\n{p['harness']}"
    args = (
        [runner, "--no-env-file", "-e", src]
        if Path(runner).name.startswith("bun")
        else [runner, "--input-type=module", "-e", src]
    )
    proc = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=15,
        env={"PATH": os.environ.get("PATH", "")},
    )
    if proc.returncode != 0:
        return Verdict(False, f"harness failed: {_harness_error(proc.stderr)}")
    return Verdict(True, "harness passed")


CHECKERS: dict[str, Checker] = {
    "none": check_none,
    "final_number": check_final_number,
    "regex": check_regex,
    "contains_all": check_contains_all,
    "exact_lines": check_exact_lines,
    "json_equals": check_json_equals,
    "html_complete": check_html_complete,
    "tool_called": check_tool_called,
    "js_harness": check_js_harness,
}


def run_check(check_type: str, params: dict[str, Any], inp: CheckInput) -> Verdict:
    try:
        checker = CHECKERS[check_type]
    except KeyError:
        return Verdict(False, f"unknown check type {check_type!r}")
    return checker(inp, params)
