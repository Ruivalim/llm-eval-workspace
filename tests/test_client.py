import httpx
import pytest

from llm_eval.core.client import (
    MAX_RETRIES,
    MAX_TOOL_ROUNDS,
    ChatResult,
    Delta,
    GenerationParams,
    LLMError,
    make_client,
    run_with_fault,
    stream_chat,
)
from llm_eval.core.tool_mock import schemas_for
from tests.fake_server import FakeServer, sse, tool_call_chunks

MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
P = GenerationParams(max_tokens=50, timeout_s=5)


def _run(server: FakeServer, tools=None, params=P):
    client = make_client("http://fake/v1", "k", params.timeout_s, http_client=server.client())
    items = list(stream_chat(client, "m", MSGS, params, tools=tools))
    deltas = [i for i in items if isinstance(i, Delta)]
    result = items[-1]
    assert isinstance(result, ChatResult)
    return deltas, result, server


def test_plain_stream_collects_content_and_usage():
    usage = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    srv = FakeServer([sse([{"role": "assistant", "content": "Hel"}, {"content": "lo"}], usage)])
    deltas, res, srv = _run(srv)
    assert [d.text for d in deltas] == ["Hel", "lo"]
    assert res.answer == "Hello"
    assert res.telemetry.tokens_out == 3 and res.telemetry.usage_from_server
    assert res.telemetry.ttft_ms <= res.telemetry.total_ms
    body = srv.body()
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["max_tokens"] == 50 and "tools" not in body
    srv2 = FakeServer([sse([{"content": "x"}])])
    _run(srv2, params=GenerationParams(timeout_s=5))
    assert "max_tokens" not in srv2.body()


def test_reasoning_split_from_content_and_counts_as_first_token():
    srv = FakeServer(
        [sse([{"reasoning_content": "think "}, {"reasoning_content": "more"}, {"content": "42"}])]
    )
    deltas, res, _ = _run(srv)
    assert [d.kind for d in deltas] == ["reasoning", "reasoning", "content"]
    assert res.reasoning == "think more" and res.answer == "42"
    assert not res.telemetry.usage_from_server
    assert res.telemetry.tokens_think > 0


def test_tool_round_trip_appends_tool_messages_and_records_calls():
    srv = FakeServer(
        [
            sse(tool_call_chunks("calculate_math", {"expression": "1284*43"})),
            sse([{"content": "It is 55212."}]),
        ]
    )
    deltas, res, srv = _run(srv, tools=schemas_for(["calculate_math"]))
    assert res.tool_calls == ["calculate_math"]
    assert res.answer == "It is 55212."
    kinds = [d.kind for d in deltas]
    assert kinds == ["tool_call", "tool_result", "content"]
    assert "55212" in deltas[1].text
    second = srv.body(1)
    roles = [m["role"] for m in second["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    assert (
        second["messages"][2]["tool_calls"][0]["function"]["arguments"]
        == '{"expression": "1284*43"}'
    )
    assert second["messages"][3]["tool_call_id"] == "call_1"
    assert srv.body(0)["tools"][0]["function"]["name"] == "calculate_math"


def test_tool_loop_stops_after_max_rounds():
    script = [sse(tool_call_chunks("get_weather", {"city": "x"}))] * (MAX_TOOL_ROUNDS + 1)
    srv = FakeServer(script)
    with pytest.raises(LLMError, match="tool loop exceeded"):
        _run(srv, tools=schemas_for(["get_weather"]))


def test_repetition_loop_cuts_the_stream_early():
    block = "The answer is seven always. "
    assert len(block) >= 24
    srv = FakeServer([sse([{"content": block}] * 40)])
    deltas, res, _ = _run(srv)
    assert res.finish_reason == "loop"
    assert len(deltas) < 40  # cut before the script ran out
    assert len(res.answer) < len(block) * 40


@pytest.mark.parametrize(
    ("text", "looped"),
    [
        (
            " ".join(
                f"Point {i}: the result depends on factor {i} and prior context." for i in range(20)
            ),
            False,
        ),
        ("The answer is seven always. " * 14, True),
        ("The answer is seven always. " * 3, False),
        ("ab" * 1000, True),  # degenerate two-char run is a loop too
        ("x" * 200, False),  # too short to call
        ("\n".join(f"def f{i}():\n    return {i}" for i in range(50)), False),
        ("\n".join(["This repeated finding line is long enough."] * 8), True),
        ("\n".join(["This repeated finding line is long enough."] * 7), False),
        (
            ("---\n" * 30)
            + "# Title here\n\n"
            + " ".join(f"Step {i} measures a different quantity entirely." for i in range(30)),
            False,
        ),
    ],
)
def test_looks_looped(text, looped):
    from llm_eval.core.client import _looks_looped

    assert _looks_looped(text) is looped


def test_malformed_tool_arguments_do_not_crash():
    chunks = [
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "c1",
                    "function": {"name": "get_weather", "arguments": "{not json"},
                }
            ]
        },
        {"finish_reason": "tool_calls"},
    ]
    srv = FakeServer([sse(chunks), sse([{"content": "ok"}])])
    _, res, _ = _run(srv, tools=schemas_for(["get_weather"]))
    assert res.tool_calls == ["get_weather"] and res.answer == "ok"


def test_malformed_chunk_maps_to_llm_error():
    srv = FakeServer([sse([{"content": "a"}], raw_tail="data: {this is not json}\n\n")])
    with pytest.raises(LLMError):
        _run(srv)


def test_mid_stream_disconnect_maps_to_llm_error():
    def boom(request):
        raise httpx.RemoteProtocolError("peer closed", request=request)

    srv = FakeServer([boom])
    with pytest.raises(LLMError, match="connection failed"):
        _run(srv)


@pytest.mark.parametrize(
    ("status", "body", "needle"),
    [
        (401, {"error": {"message": "bad key"}}, "authentication failed"),
        (404, {"error": {"message": "model 'x' not found"}}, "model 'x' not found"),
        (500, {"error": {"message": "boom"}}, "HTTP 500: boom"),
    ],
)
def test_http_status_errors_are_readable(status, body, needle):
    # 5xx is retried MAX_RETRIES times before it surfaces, so script every try.
    srv = FakeServer([httpx.Response(status, json=body) for _ in range(MAX_RETRIES + 1)])
    with pytest.raises(LLMError, match=needle):
        _run(srv)


def test_connection_refused_is_readable():
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    srv = FakeServer([refuse])
    with pytest.raises(LLMError, match="connection failed"):
        _run(srv)


def test_timeout_is_readable():
    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)

    srv = FakeServer([slow])
    with pytest.raises(LLMError, match="timeout"):
        _run(srv)


def test_fault_bad_key_and_wrong_model_change_the_request():
    srv = FakeServer([sse([{"content": "x"}]), sse([{"content": "y"}])])
    list(run_with_fault({"kind": "bad_key"}, "http://fake/v1", "good", "m", MSGS, P, srv.client()))
    assert srv.requests[0].headers["authorization"] == "Bearer invalid-key-for-fault-injection"
    list(
        run_with_fault(
            {"kind": "wrong_model"}, "http://fake/v1", "good", "m", MSGS, P, srv.client()
        )
    )
    assert srv.body(1)["model"] == "model-that-does-not-exist"


def test_fault_huge_prompt_inflates_last_message():
    srv = FakeServer([sse([{"content": "x"}])])
    list(
        run_with_fault(
            {"kind": "huge_prompt", "words": 1000},
            "http://fake/v1",
            "k",
            "m",
            MSGS,
            P,
            srv.client(),
        )
    )
    last = srv.body()["messages"][-1]["content"]
    assert last.endswith("hi") and len(last) > 10_000


def test_fault_unknown_kind_raises():
    with pytest.raises(LLMError, match="unknown fault"):
        list(run_with_fault({"kind": "nope"}, "http://fake/v1", "k", "m", MSGS, P))


def test_map_error_httpx_status_error_is_one_line():
    from llm_eval.core.client import map_error

    req = httpx.Request("GET", "http://x/v1/models")
    resp = httpx.Response(401, request=req, text="nope")
    err = map_error(httpx.HTTPStatusError("boom", request=req, response=resp))
    assert str(err) == "authentication failed (401): check the API key"
    resp = httpx.Response(503, request=req, text="down\nfor\nmaintenance")
    msg = str(map_error(httpx.HTTPStatusError("b", request=req, response=resp)))
    assert msg == "HTTP 503: down for maintenance"


def test_rate_limit_retries_then_succeeds(no_backoff_sleep):
    srv = FakeServer(
        [
            httpx.Response(429, json={"error": {"message": "slow"}}, headers={"retry-after": "3"}),
            httpx.Response(503, json={"error": {"message": "busy"}}),
            sse([{"content": "ok"}]),
        ]
    )
    _, res, _ = _run(srv)
    assert res.answer == "ok" and no_backoff_sleep == [3.0, 2.0]


def test_retry_after_is_capped_and_garbage_falls_back(no_backoff_sleep):
    srv = FakeServer(
        [
            httpx.Response(429, json={}, headers={"retry-after": "3600"}),
            httpx.Response(429, json={}, headers={"retry-after": "soon"}),
            sse([{"content": "ok"}]),
        ]
    )
    _run(srv)
    assert no_backoff_sleep == [30.0, 2.0]


def test_client_errors_and_timeouts_are_not_retried(no_backoff_sleep):
    srv = FakeServer([httpx.Response(400, json={"error": {"message": "bad"}})])
    with pytest.raises(LLMError, match="HTTP 400: bad"):
        _run(srv)
    assert len(srv.requests) == 1 and no_backoff_sleep == []


def test_gateway_reported_cost_is_summed_across_rounds():
    srv = FakeServer(
        [
            sse(
                tool_call_chunks("calculate_math", {"expression": "2*2"}),
                {"completion_tokens": 3, "cost": 0.001},
            ),
            sse([{"content": "4"}], {"completion_tokens": 2, "cost": 0.0005}),
        ]
    )
    _, res, _ = _run(srv, tools=schemas_for(["calculate_math"]))
    assert res.telemetry.cost_reported == pytest.approx(0.0015)


def test_no_reported_cost_stays_unknown():
    srv = FakeServer([sse([{"content": "hi"}], {"completion_tokens": 1})])
    _, res, _ = _run(srv)
    assert res.telemetry.cost_reported is None


def test_map_error_passes_readable_errors_through():
    from llm_eval.core.client import map_error

    err = LLMError("authentication failed (401): check the API key")
    assert map_error(err) is err
