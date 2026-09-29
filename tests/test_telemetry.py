import time

from llm_eval.core.telemetry import Stopwatch, count_tokens


def test_uses_server_usage_when_present():
    sw = Stopwatch()
    sw.first_token()
    t = sw.finish(
        "p",
        "a",
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "completion_tokens_details": {"reasoning_tokens": 5},
        },
    )
    assert (t.tokens_in, t.tokens_out, t.tokens_think) == (10, 20, 5)
    assert t.usage_from_server


def test_falls_back_to_tiktoken_without_usage():
    sw = Stopwatch()
    sw.first_token()
    t = sw.finish("hello world", "the quick brown fox jumps", reasoning="hmm")
    assert t.tokens_in == count_tokens("hello world")
    assert t.tokens_out == count_tokens("the quick brown fox jumpshmm")
    assert not t.usage_from_server


def test_ttft_before_total_and_no_first_token_means_ttft_equals_total():
    sw = Stopwatch()
    time.sleep(0.01)
    t = sw.finish("p", "")
    assert t.ttft_ms == t.total_ms >= 10
    assert t.tps == 0


def test_first_token_only_recorded_once():
    sw = Stopwatch()
    sw.first_token()
    first = sw.t_first
    time.sleep(0.005)
    sw.first_token()
    assert sw.t_first == first
