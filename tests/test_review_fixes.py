from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from llm_eval import api
from llm_eval.config import Settings
from llm_eval.core.telemetry import Telemetry
from llm_eval.db import crud
from tests.test_api import LIB


@pytest.fixture
def client(tmp_path):
    settings = Settings(
        tmp_path / "review.db",
        "http://fake/v1",
        "k",
        "en",
        tmp_path / "art",
        tmp_path / "tr",
        tmp_path / "logs",
    )
    with TestClient(api.create_app(settings, LIB)) as test_client:
        yield test_client


def _result(conn, server_id, model, locale, prompt, status, score=None):
    run = crud.new_run(conn, server_id, model, locale)
    result_id = crud.save_result(conn, run, prompt, "text", status, "", "", Telemetry())
    if score is not None:
        crud.set_human_score(conn, result_id, score)
    return result_id


def test_leaderboard_keeps_pending_human_scores_out_of_pass_rate(conn):
    server_id = crud.add_server(conn, "s", "http://x", None)
    _result(conn, server_id, "m", "en", "automatic", "PASS", 2)
    _result(conn, server_id, "m", "en", "manual", "PENDING", 10)

    row = crud.leaderboard(conn)[0]

    assert row["runs"] == 1
    assert row["pass_rate"] == 100.0
    assert row["avg_human"] == 6.0


def test_matrix_separates_server_and_locale(conn):
    one = crud.add_server(conn, "one", "http://one", None)
    two = crud.add_server(conn, "two", "http://two", None)
    en = _result(conn, one, "m", "en", "p", "PASS")
    pt = _result(conn, one, "m", "pt-BR", "p", "FAIL")
    other = _result(conn, two, "m", "en", "p", "ERROR")

    assert {row["id"] for row in crud.matrix(conn)} == {en, pt, other}
    assert [row["id"] for row in crud.matrix(conn, locale="en")] == [en, other]


def test_state_uses_independent_sqlite_connections_per_worker(client):
    state = client.app.state.state
    server_id = crud.add_server(state.conn, "parallel", "http://parallel", None)

    with ThreadPoolExecutor(max_workers=16) as pool:
        ids = list(pool.map(lambda _: crud.new_run(state.conn, server_id, "m", "en"), range(200)))

    assert len(ids) == len(set(ids)) == 200


def test_server_can_update_key_without_exposing_it(client):
    created = client.post(
        "/api/servers", json={"name": "editable", "base_url": "http://old", "api_key": "old"}
    ).json()
    response = client.put(
        f"/api/servers/{created['id']}",
        json={"name": "edited", "base_url": "http://new/", "flavor": "unsloth", "api_key": "new"},
    )

    assert response.status_code == 200
    assert response.json() == {
        **created,
        "name": "edited",
        "base_url": "http://new",
        "flavor": "unsloth",
        "has_key": True,
    }
    assert crud.list_servers(client.app.state.state.conn)[1].api_key == "new"
