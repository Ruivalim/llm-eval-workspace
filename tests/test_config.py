"""Settings come from the environment; the public repo must not ship a default
endpoint and code execution has to be opt-in."""

from __future__ import annotations

import pytest

from llm_eval.config import ROOT, load_settings


@pytest.fixture
def no_dotenv(monkeypatch):
    """load_settings calls load_dotenv, which would fill the vars from the dev
    machine's .env and hide what the env actually says."""
    monkeypatch.setattr("llm_eval.config.load_dotenv", lambda *a, **k: False)
    monkeypatch.chdir(ROOT)


def test_no_endpoint_and_no_code_exec_by_default(monkeypatch, no_dotenv):
    for name in (
        "LLM_EVAL_DEFAULT_BASE_URL",
        "LLM_EVAL_DEFAULT_API_KEY",
        "LLM_EVAL_ALLOW_CODE_EXEC",
    ):
        monkeypatch.delenv(name, raising=False)
    s = load_settings()
    assert s.default_base_url == "" and s.default_api_key is None
    assert s.allow_code_exec is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", True), ("true", True), ("YES", True), ("", False), ("0", False), ("false", False)],
)
def test_allow_code_exec_reads_a_flag(monkeypatch, no_dotenv, value, expected):
    monkeypatch.setenv("LLM_EVAL_ALLOW_CODE_EXEC", value)
    assert load_settings().allow_code_exec is expected


def test_default_server_comes_from_the_environment(monkeypatch, no_dotenv):
    monkeypatch.setenv("LLM_EVAL_DEFAULT_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_EVAL_DEFAULT_API_KEY", "k")
    monkeypatch.setenv("LLM_EVAL_DEFAULT_FLAVOR", "unsloth")
    s = load_settings()
    assert s.default_base_url == "http://localhost:11434/v1"
    assert s.default_api_key == "k" and s.default_flavor == "unsloth"
