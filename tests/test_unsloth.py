import httpx
import pytest

from llm_eval.core import unsloth


@pytest.mark.parametrize(
    ("base", "want"),
    [
        ("https://h/v1", "https://h"),
        ("https://h/v1/", "https://h"),
        ("https://h", "https://h"),
        ("https://h/api/v1", "https://h/api"),
    ],
)
def test_origin_strips_only_trailing_v1(base, want):
    assert unsloth._origin(base) == want


class _Server:
    def __init__(self, loaded_after: int):
        self.calls = 0
        self.loaded_after = loaded_after
        self.load_bodies = []

    def handler(self, req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/inference/load":
            self.load_bodies.append(req.content)
            return httpx.Response(200, json={"ok": True})
        if req.url.path == "/v1/models":
            self.calls += 1
            loaded = "b" if self.calls >= self.loaded_after else "a"
            return httpx.Response(
                200,
                json={"data": [{"id": m, "loaded": m == loaded} for m in ("a", "b")]},
            )
        return httpx.Response(404)

    def client(self):
        return httpx.Client(transport=httpx.MockTransport(self.handler))


def test_load_posts_to_origin_and_polls_until_match():
    srv = _Server(loaded_after=3)
    slept = []
    unsloth.load("https://h/v1", "k", "b", client=srv.client(), sleep=slept.append, poll_s=1)
    assert srv.load_bodies == [b'{"model_path":"b"}']
    assert srv.calls == 3
    assert slept == [1, 1]


def test_load_times_out_using_injected_clock():
    srv = _Server(loaded_after=10**9)
    t = [0.0]

    def clock():
        return t[0]

    def sleep(s):
        t[0] += s

    with pytest.raises(TimeoutError, match="not active after 10"):
        unsloth.load(
            "https://h/v1",
            None,
            "b",
            wait_s=10,
            poll_s=4,
            client=srv.client(),
            sleep=sleep,
            clock=clock,
        )
    assert srv.calls == 4  # t=0,4,8,12 -> deadline reached after the 4th poll


def test_load_raises_on_http_error_before_polling():
    def handler(req):
        return httpx.Response(500, json={"error": "nope"})

    with pytest.raises(httpx.HTTPStatusError):
        unsloth.load(
            "https://h/v1", None, "b", client=httpx.Client(transport=httpx.MockTransport(handler))
        )


def test_active_returns_none_when_nothing_loaded():
    def handler(req):
        return httpx.Response(200, json={"data": [{"id": "a", "loaded": False}]})

    assert (
        unsloth.active(
            "https://h/v1", None, client=httpx.Client(transport=httpx.MockTransport(handler))
        )
        is None
    )
