import json

import pytest

from llm_eval.core import anthropic_client
from llm_eval.core.anthropic_client import (
    DEFAULT_MAX_TOKENS,
    make_client,
    normalize_base_url,
    stream_messages,
    to_anthropic_tools,
)
from llm_eval.core.client import (
    MAX_RETRIES,
    CancelHook,
    ChatResult,
    Delta,
    GenerationParams,
    LLMError,
    Stopped,
)
from llm_eval.core.tool_mock import schemas_for
from tests.fake_anthropic import (
    FakeAnthropic,
    error,
    messages_sse,
    text_block,
    thinking_block,
    tool_block,
)

MSGS = [{"role": "system", "content": "be terse"}, {"role": "user", "content": "2+2?"}]
P = GenerationParams(temperature=0.3, top_p=0.9, timeout_s=5)


@pytest.fixture(autouse=True)
def _fresh_caps(monkeypatch):
    monkeypatch.setattr(anthropic_client, "_output_caps", {})


def _run(srv: FakeAnthropic, params=P, tools=None, hook=None, model="claude-test", key="sk-test"):
    client = make_client("https://api.anthropic.com/v1", key, params.timeout_s, srv.client())
    items = list(stream_messages(client, model, MSGS, params, tools=tools, hook=hook))
    result = items[-1]
    assert isinstance(result, ChatResult)
    return [i for i in items if isinstance(i, Delta)], result


def test_text_stream_body_and_usage():
    srv = FakeAnthropic(
        [messages_sse([text_block("four")], usage_in=10, usage_out=3, cache_read=5)],
        output_caps={"claude-test": 8192},
    )
    deltas, res = _run(srv)
    assert "".join(d.text for d in deltas if d.kind == "content") == "four"
    assert res.answer == "four" and res.finish_reason == "stop"
    # cache reads are input the model processed
    assert res.telemetry.tokens_in == 15 and res.telemetry.tokens_out == 3
    assert res.telemetry.usage_from_server
    body = srv.body()
    assert body["system"] == "be terse"
    assert body["messages"] == [{"role": "user", "content": "2+2?"}]
    assert body["stream"] is True
    # current Claude models 400 on sampling params: never sent
    assert "temperature" not in body and "top_p" not in body
    # no prompt cap: the model's own output cap from the Models API
    assert body["max_tokens"] == 8192


def test_output_cap_is_bounded_and_falls_back():
    big = FakeAnthropic([messages_sse([text_block("x")])], output_caps={"claude-test": 128_000})
    _run(big)
    assert big.body()["max_tokens"] == DEFAULT_MAX_TOKENS
    unknown = FakeAnthropic([messages_sse([text_block("x")])], output_caps={})
    _run(unknown, model="claude-missing")
    assert unknown.body()["max_tokens"] == DEFAULT_MAX_TOKENS


def test_prompt_max_tokens_wins_over_lookup():
    srv = FakeAnthropic([messages_sse([text_block("x")])], output_caps={"claude-test": 8192})
    _run(srv, params=GenerationParams(max_tokens=300, timeout_s=5))
    assert srv.body()["max_tokens"] == 300


def test_no_system_message_omits_system():
    srv = FakeAnthropic([messages_sse([text_block("x")])])
    client = make_client("https://api.anthropic.com", "k", 5, srv.client())
    list(stream_messages(client, "m", [{"role": "user", "content": "hi"}], P))
    assert "system" not in srv.body()


def test_thinking_streams_as_reasoning():
    srv = FakeAnthropic(
        [messages_sse([thinking_block("add them"), text_block("4")], thinking_tokens=9)]
    )
    deltas, res = _run(srv)
    assert [d.kind for d in deltas][0] == "reasoning"
    assert res.reasoning == "add them" and res.answer == "4"
    assert res.telemetry.tokens_think == 9


def test_tool_loop_echoes_the_whole_assistant_turn():
    srv = FakeAnthropic(
        [
            messages_sse(
                [
                    thinking_block("use the tool"),
                    tool_block("calculate_math", {"expression": "6*7"}),
                ],
                stop_reason="tool_use",
            ),
            messages_sse([text_block("42")]),
        ]
    )
    deltas, res = _run(srv, tools=schemas_for(["calculate_math"]))
    assert res.answer == "42" and res.tool_calls == ["calculate_math"]
    assert [d.kind for d in deltas if d.kind.startswith("tool")] == ["tool_call", "tool_result"]
    first, second = srv.body(0), srv.body(1)
    assert first["tools"][0]["name"] == "calculate_math"
    assert "input_schema" in first["tools"][0] and "function" not in first["tools"][0]
    assistant, tool_turn = second["messages"][1], second["messages"][2]
    assert assistant["role"] == "assistant"
    # the thinking block goes back with its signature, untouched
    assert assistant["content"][0]["type"] == "thinking"
    assert assistant["content"][0]["signature"] == "sig-1"
    assert assistant["content"][1] == {
        "type": "tool_use",
        "id": "toolu_1",
        "name": "calculate_math",
        "input": {"expression": "6*7"},
    }
    assert tool_turn["role"] == "user"
    assert tool_turn["content"][0]["type"] == "tool_result"
    assert tool_turn["content"][0]["tool_use_id"] == "toolu_1"
    assert "42" in tool_turn["content"][0]["content"]
    # usage adds up across rounds
    assert res.telemetry.tokens_out == 14


def test_parallel_tool_results_go_in_one_user_message():
    srv = FakeAnthropic(
        [
            messages_sse(
                [
                    tool_block("calculate_math", {"expression": "1+1"}, "toolu_a"),
                    tool_block("calculate_math", {"expression": "2+2"}, "toolu_b"),
                ],
                stop_reason="tool_use",
            ),
            messages_sse([text_block("2 and 4")]),
        ]
    )
    _run(srv, tools=schemas_for(["calculate_math"]))
    turn = srv.body(1)["messages"][2]
    assert [c["tool_use_id"] for c in turn["content"]] == ["toolu_a", "toolu_b"]


@pytest.mark.parametrize(
    ("stop", "finish"),
    [
        ("max_tokens", "length"),
        ("refusal", "refusal"),
        ("end_turn", "stop"),
        # Xiaomi MiMo's own stop reasons
        ("repetition_truncation", "loop"),
        ("content_filter", "refusal"),
        ("something_new", "something_new"),
    ],
)
def test_stop_reason_maps_to_finish(stop, finish):
    srv = FakeAnthropic([messages_sse([text_block("partial")], stop_reason=stop)])
    _, res = _run(srv)
    assert res.finish_reason == finish


def test_max_tokens_with_tool_use_does_not_run_the_tool():
    srv = FakeAnthropic(
        [
            messages_sse(
                [tool_block("calculate_math", {"expression": "6*7"})], stop_reason="max_tokens"
            )
        ]
    )
    deltas, res = _run(srv, tools=schemas_for(["calculate_math"]))
    assert res.finish_reason == "length" and res.tool_calls == []
    assert len(srv.requests) == 1


def test_repetition_loop_is_cut():
    line = "the same sentence repeated over and over again.\n"
    srv = FakeAnthropic([messages_sse([{"type": "text", "pieces": [line] * 60}])])
    _, res = _run(srv)
    assert res.finish_reason == "loop"
    assert len(res.answer) < len(line) * 60


@pytest.mark.parametrize(
    ("status", "needle"),
    [(401, "authentication failed"), (404, "not found \\(404\\)"), (400, "HTTP 400: bad field")],
)
def test_errors_are_one_readable_line(status, needle):
    srv = FakeAnthropic([error(status, "bad field")])
    with pytest.raises(LLMError, match=needle) as info:
        _run(srv)
    assert "\n" not in str(info.value)
    assert len(srv.requests) == 1  # client errors are never retried


def test_rate_limit_is_retried_with_retry_after(no_backoff_sleep):
    srv = FakeAnthropic(
        [error(429, "slow down", {"retry-after": "3"}), messages_sse([text_block("ok")])]
    )
    _, res = _run(srv)
    assert res.answer == "ok"
    assert no_backoff_sleep == [3.0]


def test_overloaded_gives_up_after_max_retries(no_backoff_sleep):
    srv = FakeAnthropic([error(529, "overloaded") for _ in range(MAX_RETRIES + 1)])
    with pytest.raises(LLMError, match="HTTP 529: overloaded"):
        _run(srv)
    assert len(srv.requests) == MAX_RETRIES + 1
    assert no_backoff_sleep == [1.0, 2.0]


def test_stop_during_backoff_sends_nothing_more(monkeypatch):
    hook = CancelHook()

    def stop_now(seconds, h):
        h.cancel()
        return True

    monkeypatch.setattr("llm_eval.core.client._sleep", stop_now)
    srv = FakeAnthropic([error(429, "slow"), messages_sse([text_block("never")])])
    with pytest.raises(Stopped):
        _run(srv, hook=hook)
    assert len(srv.requests) == 1


def test_stop_mid_stream_raises_stopped():
    hook = CancelHook()
    srv = FakeAnthropic([messages_sse([text_block("a long answer")])])
    client = make_client("https://api.anthropic.com", "k", 5, srv.client())
    gen = stream_messages(client, "m", MSGS, P, hook=hook)
    assert isinstance(next(gen), Delta)
    hook.cancel()
    with pytest.raises(Stopped):
        list(gen)


def test_missing_key_never_falls_back_to_the_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env-must-not-be-sent")
    srv = FakeAnthropic([messages_sse([text_block("x")])])
    _run(srv, key=None)
    sent = srv.requests[0].headers.get("x-api-key")
    assert sent != "sk-from-env-must-not-be-sent"


def test_base_url_accepts_with_or_without_v1():
    assert normalize_base_url("https://api.anthropic.com/v1/") == "https://api.anthropic.com"
    assert normalize_base_url("https://api.anthropic.com") == "https://api.anthropic.com"
    assert normalize_base_url("") == "https://api.anthropic.com"
    srv = FakeAnthropic([messages_sse([text_block("x")])])
    _run(srv)
    assert srv.requests[0].url.path == "/v1/messages"


def test_tool_schema_conversion():
    [tool] = to_anthropic_tools(schemas_for(["get_weather"]))
    assert tool["name"] == "get_weather"
    assert tool["input_schema"]["type"] == "object"
    assert json.dumps(tool)  # plain JSON, nothing SDK-specific


def test_list_models():
    srv = FakeAnthropic([], output_caps={"claude-a": 1, "claude-b": 2})
    client = make_client("https://api.anthropic.com", "k", 5, srv.client())
    assert [m["id"] for m in anthropic_client.list_models(client)] == ["claude-a", "claude-b"]


@pytest.mark.parametrize(
    ("pasted", "root"),
    [
        ("https://api.xiaomimimo.com/anthropic", "https://api.xiaomimimo.com/anthropic"),
        ("https://api.xiaomimimo.com/anthropic/v1", "https://api.xiaomimimo.com/anthropic"),
        (
            "https://api.xiaomimimo.com/anthropic/v1/messages",
            "https://api.xiaomimimo.com/anthropic",
        ),
    ],
)
def test_gateway_urls_from_docs_are_accepted(pasted, root):
    assert normalize_base_url(pasted) == root


def test_gateway_gets_bearer_too_and_anthropic_does_not():
    gateway = FakeAnthropic([messages_sse([text_block("x")])])
    client = make_client("https://api.xiaomimimo.com/anthropic", "mimo-key", 5, gateway.client())
    list(stream_messages(client, "mimo-v2.5", MSGS, GenerationParams(max_tokens=10, timeout_s=5)))
    sent = gateway.requests[0]
    assert sent.url.path == "/anthropic/v1/messages"
    assert sent.headers["authorization"] == "Bearer mimo-key"
    assert sent.headers["x-api-key"] == "mimo-key"

    official = FakeAnthropic([messages_sse([text_block("x")])])
    client = make_client("https://api.anthropic.com", "sk-ant", 5, official.client())
    list(stream_messages(client, "claude-x", MSGS, GenerationParams(max_tokens=10, timeout_s=5)))
    assert "authorization" not in official.requests[0].headers
    assert official.requests[0].headers["x-api-key"] == "sk-ant"


def _gateway(routes):
    """A gateway mounted at /anthropic; `routes` maps a path to a response."""
    import httpx2

    seen = []

    def handler(request):
        seen.append(request.url.path)
        if request.url.path in routes:
            return routes[request.url.path]
        return httpx2.Response(404, text="<html><body><h1>404 Not Found</h1></body></html>")

    return httpx2.Client(transport=httpx2.MockTransport(handler)), seen


def test_gateway_without_models_route_lists_from_its_root():
    import httpx2

    http, seen = _gateway(
        {
            "/v1/models": httpx2.Response(
                200,
                json={"object": "list", "data": [{"id": "mimo-v2.5"}, {"id": "mimo-v2.6-flash"}]},
            )
        }
    )
    client = make_client("https://api.xiaomimimo.com/anthropic", "k", 5, http)
    assert anthropic_client.list_models(client) == [{"id": "mimo-v2.5"}, {"id": "mimo-v2.6-flash"}]
    assert seen == ["/anthropic/v1/models", "/v1/models"]


def test_gateway_with_no_listing_anywhere_says_so_without_html():
    http, _ = _gateway({})
    client = make_client("https://gw.example/anthropic", "k", 5, http)
    with pytest.raises(LLMError) as info:
        anthropic_client.list_models(client)
    assert "404" in str(info.value) and "<html" not in str(info.value)


def test_official_host_never_probes_the_root():
    http, seen = _gateway({})
    client = make_client("https://api.anthropic.com", "k", 5, http)
    with pytest.raises(LLMError):
        anthropic_client.list_models(client)
    assert seen == ["/v1/models"]


def test_trace_records_the_echoed_turn_as_json(tmp_path):
    from llm_eval.core.trace import Trace, read_trace

    srv = FakeAnthropic(
        [
            messages_sse(
                [thinking_block("plan"), tool_block("calculate_math", {"expression": "1+1"})],
                stop_reason="tool_use",
            ),
            messages_sse([text_block("2")]),
        ]
    )
    client = make_client("https://api.anthropic.com", "k", 5, srv.client())
    with Trace.start(tmp_path, "p", "m") as tr:
        list(stream_messages(client, "m", MSGS, P, tools=schemas_for(["calculate_math"]), log=tr))
        path = tr.path
    second = [e for e in read_trace(path) if e["ev"] == "request"][1]
    echoed = second["messages"][1]["content"]
    assert [b["type"] for b in echoed] == ["thinking", "tool_use"]
    assert echoed[0]["signature"] == "sig-1"
