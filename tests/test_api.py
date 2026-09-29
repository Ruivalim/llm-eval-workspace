"""API tests: TestClient + the SSE fake server for the model side."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from llm_eval import api
from llm_eval.config import Settings
from llm_eval.core import engine
from llm_eval.core.prompts import Library, Prompt
from tests.fake_server import FakeServer, sse

LIB = Library(
    prompts=[
        Prompt.model_validate(
            {
                "id": "sum",
                "category": "logic",
                "eval_type": "text",
                "check": {"type": "final_number", "expected": 4},
                "locales": {"en": {"name": "Sum", "prompt": "2+2?"}},
            }
        ),
        Prompt.model_validate(
            {
                "id": "page",
                "category": "coding-ui",
                "eval_type": "render_html",
                "check": {"type": "html_complete"},
                "locales": {"en": {"name": "Page", "prompt": "make a page"}},
            }
        ),
    ]
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    settings = Settings(
        tmp_path / "t.db",
        "http://fake/v1",
        "k",
        "en",
        tmp_path / "art",
        tmp_path / "tr",
        tmp_path / "logs",
        default_flavor="unsloth",
    )
    app = api.create_app(settings, LIB)
    with TestClient(app) as c:
        yield c


def _events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        ev = data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                ev = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        out.append((ev, data))
    return out


def _use_fake(monkeypatch, fake: FakeServer):
    real = engine.make_client

    def patched(base_url, api_key, timeout_s, http_client=None):
        return real(base_url, api_key, timeout_s, http_client=fake.client())

    monkeypatch.setattr(engine, "make_client", patched)


def test_meta_lists_prompts_and_seeded_server(client):
    m = client.get("/api/meta").json()
    assert [p["id"] for p in m["prompts"]] == ["sum", "page"]
    assert m["prompts"][0]["auto"] is True
    assert "pt-BR" in m["locales"]
    srv = client.get("/api/servers").json()
    assert len(srv) == 1 and srv[0]["flavor"] == "unsloth" and srv[0]["has_key"] is True
    assert "api_key" not in srv[0]


def test_server_crud_and_conflicts(client):
    r = client.post("/api/servers", json={"name": "x", "base_url": "http://x/v1/"})
    assert r.status_code == 201 and r.json()["base_url"] == "http://x/v1"
    assert (
        client.post("/api/servers", json={"name": "x", "base_url": "http://y"}).status_code == 409
    )
    assert client.post("/api/servers", json={"name": "", "base_url": "http://y"}).status_code == 422
    assert (
        client.post("/api/servers", json={"name": "z", "base_url": "u", "flavor": "no"}).status_code
        == 422
    )
    sid = r.json()["id"]
    assert client.delete(f"/api/servers/{sid}").status_code == 204
    assert client.delete(f"/api/servers/{sid}").status_code == 404


def test_models_error_is_502_with_readable_message(client, monkeypatch):
    def refuse(req):
        raise httpx.ConnectError("refused", request=req)

    _use_fake(monkeypatch, FakeServer([refuse]))
    monkeypatch.setattr(api, "make_client", engine.make_client)
    client.post("/api/servers", json={"name": "o", "base_url": "http://o/v1"})
    sid = client.get("/api/servers").json()[-1]["id"]
    r = client.get(f"/api/servers/{sid}/models")
    assert r.status_code == 502 and "connection failed" in r.json()["detail"]


def test_evaluate_streams_deltas_then_outcome_and_saves(client, monkeypatch):
    fake = FakeServer(
        [
            sse(
                [{"content": "It is\n"}, {"content": "4"}],
                {"prompt_tokens": 2, "completion_tokens": 3},
            )
        ]
    )
    _use_fake(monkeypatch, fake)
    sid = client.get("/api/servers").json()[0]["id"]
    with client.stream(
        "POST",
        "/api/evaluate",
        json={"server_id": sid, "model": "m", "prompt_id": "sum", "system_override": "terse"},
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = _events(r.read().decode())
    kinds = [e for e, _ in events]
    assert kinds == ["start", "delta", "delta", "outcome"]
    out = events[-1][1]
    assert (
        out["status"] == "PASS"
        and out["answer"] == "It is\n4"
        and out["telemetry"]["tokens_out"] == 3
    )
    assert fake.body()["messages"][0] == {"role": "system", "content": "terse"}
    hist = client.get("/api/history").json()
    assert hist[0]["id"] == out["result_id"] and hist[0]["status"] == "PASS"
    assert client.get(f"/api/results/{out['result_id']}").json()["answer"] == "It is\n4"
    assert client.get("/api/history", params={"status": "FAIL"}).json() == []
    lb = client.get("/api/leaderboard").json()
    assert lb[0]["model"] == "m" and lb[0]["pass_rate"] == 100.0
    assert client.get("/api/matrix").json()[0]["prompt_id"] == "sum"


def test_evaluate_records_how_many_requests_ran_at_once(client, monkeypatch):
    _use_fake(
        monkeypatch,
        FakeServer([sse([{"content": "4"}]), sse([{"content": "4"}])]),
    )
    body = {"server_id": 1, "model": "m", "prompt_id": "sum"}
    client.post("/api/evaluate", json=body)  # omitted: sequential
    client.post("/api/evaluate", json={**body, "parallel": 4})
    hist = client.get("/api/history").json()
    assert [h["parallel"] for h in hist] == [4, 1]
    assert client.get(f"/api/results/{hist[0]['id']}").json()["parallel"] == 4
    for bad in (0, 99):
        r = client.post("/api/evaluate", json={**body, "parallel": bad})
        assert r.status_code == 422


def test_evaluate_unknown_prompt_or_server(client):
    assert (
        client.post(
            "/api/evaluate", json={"server_id": 1, "model": "m", "prompt_id": "nope"}
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/api/evaluate", json={"server_id": 99, "model": "m", "prompt_id": "sum"}
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/api/evaluate", json={"server_id": 1, "model": "m", "prompt_id": "sum", "locale": "xx"}
        ).status_code
        == 422
    )


def test_evaluate_error_becomes_error_outcome(client, monkeypatch):
    def refuse(req):
        raise httpx.ConnectError("refused", request=req)

    _use_fake(monkeypatch, FakeServer([refuse]))
    r = client.post("/api/evaluate", json={"server_id": 1, "model": "m", "prompt_id": "sum"})
    events = _events(r.text)
    assert events[-1][0] == "outcome" and events[-1][1]["status"] == "ERROR"
    assert "connection failed" in events[-1][1]["error"]


def test_artifact_served_and_path_traversal_blocked(client, monkeypatch):
    html = "<!DOCTYPE html><html><body>x</body></html>"
    _use_fake(monkeypatch, FakeServer([sse([{"content": html}])]))
    r = client.post("/api/evaluate", json={"server_id": 1, "model": "m", "prompt_id": "page"})
    out = _events(r.text)[-1][1]
    assert out["artifact"].endswith(".html")
    a = client.get(f"/api/artifacts/{out['artifact']}")
    assert a.status_code == 200 and a.text == html
    assert a.headers["content-security-policy"] == "sandbox allow-scripts"
    assert client.get("/api/artifacts/../pyproject.toml").status_code in (400, 404)
    assert client.get("/api/artifacts/missing.html").status_code == 404


def test_score_validation_and_upsert(client, monkeypatch):
    _use_fake(monkeypatch, FakeServer([sse([{"content": "4"}])]))
    r = client.post("/api/evaluate", json={"server_id": 1, "model": "m", "prompt_id": "sum"})
    rid = _events(r.text)[-1][1]["result_id"]
    assert client.post(f"/api/results/{rid}/score", json={"score": 11}).status_code == 422
    assert client.post("/api/results/999/score", json={"score": 5}).status_code == 404
    assert (
        client.post(f"/api/results/{rid}/score", json={"score": 7, "comment": "ok"}).status_code
        == 200
    )
    assert client.post(f"/api/results/{rid}/score", json={"score": 9}).status_code == 200
    assert client.get(f"/api/results/{rid}").json()["human_score"] == 9


def test_spa_fallback_never_masks_api_404(monkeypatch, tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>app</html>")
    monkeypatch.setattr(api, "WEB_DIST", dist)
    app = api.create_app(
        Settings(tmp_path / "u.db", "http://fake/v1", "k", "en", tmp_path / "a"), LIB
    )
    with TestClient(app) as c:
        assert c.get("/").text == "<html>app</html>"
        assert c.get("/results").text == "<html>app</html>"
        assert c.get("/api/nope").status_code == 404
        assert c.get("/api/artifacts/../pyproject.toml").status_code == 404


def test_spa_catch_all_never_serves_files_outside_dist(monkeypatch, tmp_path):
    """The catch-all is a SPA fallback, not a file server: a path that escapes
    web/dist has to return the app shell, in both the raw and the encoded form."""
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>app</html>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    sentinel = tmp_path / "secret.txt"
    sentinel.write_text("top secret")

    # Raw `../` cannot reach the server through httpx (the client normalizes it
    # away), so the resolver is tested directly for that form.
    assert api.spa_target(dist, "../../etc/hostname") is None
    assert api.spa_target(dist, "") is None
    assert api.spa_target(dist, str(sentinel)) is None  # absolute path
    assert api.spa_target(dist, "assets/app.js") == (dist / "assets" / "app.js").resolve()
    assert api.spa_target(dist, "..%2fsecret.txt") is None  # not decoded twice

    monkeypatch.setattr(api, "WEB_DIST", dist)
    app = api.create_app(
        Settings(tmp_path / "u.db", "http://fake/v1", "k", "en", tmp_path / "a"), LIB
    )
    with TestClient(app) as c:
        for path in ("/..%2f..%2fsecret.txt", "/%2e%2e/%2e%2e/secret.txt", "/..%2fsecret.txt"):
            r = c.get(path)
            assert r.status_code == 200 and r.text == "<html>app</html>", path
        assert c.get("/assets/app.js").text == "console.log(1)"


def test_ui_is_gzipped_and_sse_is_not(monkeypatch, tmp_path):
    """The shell and the bundle go out compressed (the bundle is ~340 kB, measured
    under throttling); /api does not, so Server-Sent Events keep flushing."""
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>app</html>" + "<!-- pad -->" * 200)
    monkeypatch.setattr(api, "WEB_DIST", dist)
    app = api.create_app(
        Settings(tmp_path / "g.db", "http://fake/v1", "k", "en", tmp_path / "a"), LIB
    )
    _use_fake(monkeypatch, FakeServer([sse([{"content": "4"}])]))
    with TestClient(app) as c:
        r = c.get("/", headers={"Accept-Encoding": "gzip"})
        assert r.headers.get("content-encoding") == "gzip"
        assert r.text.startswith("<html>app</html>")
        s = c.post("/api/evaluate", json={"server_id": 1, "model": "m", "prompt_id": "sum"})
        assert "content-encoding" not in s.headers
        assert s.headers["content-type"].startswith("text/event-stream")


def test_trace_endpoint_returns_events_and_raw(client, monkeypatch):
    _use_fake(monkeypatch, FakeServer([sse([{"content": "4"}])]))
    r = client.post("/api/evaluate", json={"server_id": 1, "model": "m", "prompt_id": "sum"})
    out = _events(r.text)[-1][1]
    assert out["trace"] is True
    events = client.get(f"/api/results/{out['result_id']}/trace").json()
    assert [e["ev"] for e in events][:2] == ["evaluate", "request"] and events[-1][
        "ev"
    ] == "outcome"
    raw = client.get(f"/api/results/{out['result_id']}/trace", params={"raw": 1})
    assert raw.status_code == 200 and raw.text.count("\n") == len(events)
    assert client.get("/api/results/999/trace").status_code == 404
    assert client.get("/api/history").json()[0]["trace"] is True


def test_settings_roundtrip_and_injection(client, monkeypatch):
    assert client.get("/api/settings").json() == {"global_system_prompt": ""}
    r = client.put("/api/settings", json={"global_system_prompt": "Reply in pt-BR."})
    assert r.status_code == 200 and r.json()["global_system_prompt"] == "Reply in pt-BR."
    fake = FakeServer([sse([{"content": "4"}])])
    _use_fake(monkeypatch, fake)
    client.post("/api/evaluate", json={"server_id": 1, "model": "m", "prompt_id": "sum"})
    assert fake.body()["messages"][0] == {"role": "system", "content": "Reply in pt-BR."}


def test_delete_one_result_removes_row_artifact_and_trace(client, monkeypatch):
    html = "<!DOCTYPE html><html><body>x</body></html>"
    _use_fake(monkeypatch, FakeServer([sse([{"content": html}])]))
    r = client.post("/api/evaluate", json={"server_id": 1, "model": "m", "prompt_id": "page"})
    out = _events(r.text)[-1][1]
    rid = out["result_id"]
    assert client.get(f"/api/artifacts/{out['artifact']}").status_code == 200
    assert client.get(f"/api/results/{rid}/trace").status_code == 200

    assert client.delete(f"/api/results/{rid}").status_code == 204
    assert client.delete(f"/api/results/{rid}").status_code == 404
    assert client.get(f"/api/artifacts/{out['artifact']}").status_code == 404
    assert client.get(f"/api/results/{rid}/trace").status_code == 404
    assert client.get("/api/history").json() == []
    assert client.get("/api/matrix").json() == []


def test_clear_results_all_and_filtered(client, monkeypatch):
    _use_fake(monkeypatch, FakeServer([sse([{"content": "4"}]), sse([{"content": "5"}])]))
    for _ in range(2):
        client.post("/api/evaluate", json={"server_id": 1, "model": "m", "prompt_id": "sum"})
    assert len(client.get("/api/history").json()) == 2

    # Filter that matches nothing deletes nothing, and says so.
    assert client.delete("/api/results", params={"prompt_id": "nope"}).json() == {"deleted": 0}
    assert len(client.get("/api/history").json()) == 2

    # Zero-arg call is the "clear everything" path.
    assert client.delete("/api/results").json() == {"deleted": 2}
    assert client.get("/api/history").json() == []
    assert client.get("/api/leaderboard").json() == []
    assert client.get("/api/matrix").json() == []
    # Prompts and the server survive.
    assert client.get("/api/meta").json()["prompts"]
    assert len(client.get("/api/servers").json()) == 1


# ---- anthropic servers and prices ---------------------------------------------


def _use_fake_anthropic(monkeypatch, fake):
    from llm_eval.core import anthropic_client

    real = anthropic_client.make_client

    def patched(base_url, api_key, timeout_s, http_client=None):
        return real(base_url, api_key, timeout_s, http_client=fake.client())

    monkeypatch.setattr(anthropic_client, "make_client", patched)
    monkeypatch.setattr(anthropic_client, "_output_caps", {})


def test_flavor_is_validated_and_anthropic_is_accepted(client):
    r = client.post(
        "/api/servers",
        json={"name": "c", "base_url": "https://api.anthropic.com", "flavor": "anthropic"},
    )
    assert r.status_code == 201 and r.json()["flavor"] == "anthropic"
    r = client.post("/api/servers", json={"name": "x", "base_url": "http://x", "flavor": "gemini"})
    assert r.status_code == 422 and "anthropic" in r.json()["detail"]


def test_anthropic_models_listed(client, monkeypatch):
    from tests.fake_anthropic import FakeAnthropic

    _use_fake_anthropic(monkeypatch, FakeAnthropic([], output_caps={"claude-haiku-4-5": 1}))
    sid = client.post(
        "/api/servers",
        json={
            "name": "c",
            "base_url": "https://api.anthropic.com",
            "api_key": "k",
            "flavor": "anthropic",
        },
    ).json()["id"]
    assert client.get(f"/api/servers/{sid}/models").json() == [
        {"id": "claude-haiku-4-5", "loaded": False}
    ]


def test_evaluate_anthropic_outcome_carries_cost(client, monkeypatch):
    from tests.fake_anthropic import FakeAnthropic, messages_sse, text_block

    fake = FakeAnthropic(
        [messages_sse([text_block("4")], usage_in=1_000_000, usage_out=100_000)],
        output_caps={"claude-sonnet-5-5": 64_000},
    )
    _use_fake_anthropic(monkeypatch, fake)
    sid = client.post(
        "/api/servers",
        json={
            "name": "c",
            "base_url": "https://api.anthropic.com",
            "api_key": "k",
            "flavor": "anthropic",
        },
    ).json()["id"]
    with client.stream(
        "POST",
        "/api/evaluate",
        json={"server_id": sid, "model": "claude-sonnet-5-5", "prompt_id": "sum"},
    ) as r:
        events = _events(r.read().decode())
    out = events[-1][1]
    assert out["status"] == "PASS"
    assert out["cost_usd"] == pytest.approx(3.0) and out["cost_source"] == "builtin"
    [row] = client.get("/api/history").json()
    assert row["cost_usd"] == pytest.approx(3.0)
    [lb] = client.get("/api/leaderboard").json()
    assert lb["total_cost"] == pytest.approx(3.0) and lb["priced"] == 1


def test_prices_crud_and_backfill(client, monkeypatch):
    fake = FakeServer(
        [sse([{"content": "4"}], {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000})]
    )
    _use_fake(monkeypatch, fake)
    sid = client.get("/api/servers").json()[0]["id"]
    with client.stream(
        "POST",
        "/api/evaluate",
        json={"server_id": sid, "model": "deepseek-chat", "prompt_id": "sum"},
    ) as r:
        out = _events(r.read().decode())[-1][1]
    assert out["cost_usd"] is None  # nobody priced it yet

    listing = client.get("/api/prices").json()
    assert listing["prices"] == [] and listing["builtin_as_of"]
    assert any(b["model"] == "claude-opus-5-5" for b in listing["builtin"])

    r = client.put(
        "/api/prices",
        json={"model": " deepseek-chat ", "input_per_mtok": 0.27, "output_per_mtok": 1.1},
    )
    assert r.json() == {"model": "deepseek-chat", "filled": 1}
    assert client.get("/api/history").json()[0]["cost_usd"] == pytest.approx(1.37)
    assert client.get("/api/prices").json()["prices"][0]["model"] == "deepseek-chat"

    for bad in (
        {"model": "m", "input_per_mtok": -1, "output_per_mtok": 1},
        {"model": "", "input_per_mtok": 1, "output_per_mtok": 1},
    ):
        assert client.put("/api/prices", json=bad).status_code == 422

    assert client.delete("/api/prices", params={"model": "deepseek-chat"}).status_code == 204
    assert client.delete("/api/prices", params={"model": "deepseek-chat"}).status_code == 404
    assert client.get("/api/history").json()[0]["cost_usd"] == pytest.approx(1.37)


def test_anthropic_models_error_is_one_clean_line(client, monkeypatch):
    from tests.fake_anthropic import FakeAnthropic, error

    class Refusing(FakeAnthropic):
        def handler(self, request):
            return error(401, "invalid x-api-key")

    _use_fake_anthropic(monkeypatch, Refusing([]))
    sid = client.post(
        "/api/servers",
        json={
            "name": "c",
            "base_url": "https://api.anthropic.com",
            "api_key": "bad",
            "flavor": "anthropic",
        },
    ).json()["id"]
    r = client.get(f"/api/servers/{sid}/models")
    assert r.status_code == 502
    assert r.json()["detail"] == "authentication failed (401): check the API key"


def test_evaluate_records_effort_and_rejects_unknown_levels(client, monkeypatch):
    fake = FakeServer([sse([{"content": "4"}], {"prompt_tokens": 1, "completion_tokens": 1})])
    _use_fake(monkeypatch, fake)
    sid = client.get("/api/servers").json()[0]["id"]
    bad = client.post(
        "/api/evaluate",
        json={"server_id": sid, "model": "m", "prompt_id": "sum", "effort": "turbo"},
    )
    assert bad.status_code == 422
    with client.stream(
        "POST",
        "/api/evaluate",
        json={"server_id": sid, "model": "m", "prompt_id": "sum", "effort": "off"},
    ) as r:
        r.read()
    assert fake.body()["thinking"] == {"type": "disabled"}
    [row] = client.get("/api/history").json()
    assert row["effort"] == "off"
    assert client.get(f"/api/results/{row['id']}").json()["effort"] == "off"
    [lb] = client.get("/api/leaderboard").json()
    assert lb["effort"] == "off"
