"""Cost per evaluation: price resolution, the DB side of prices, and the
engine and API carrying the number through."""

import sqlite3

import pytest

from llm_eval.core import pricing
from llm_eval.core.pricing import Price, builtin_price, cost_of, resolve
from llm_eval.core.telemetry import Telemetry
from llm_eval.db import crud


def _t(tokens_in=1_000_000, tokens_out=1_000_000, reported=None):
    return Telemetry(tokens_in=tokens_in, tokens_out=tokens_out, cost_reported=reported)


def test_builtin_matches_exact_and_dated_ids_only():
    assert builtin_price("claude-opus-5-5") == Price(4.0, 20.0, "builtin")
    assert builtin_price("claude-haiku-4-5-20251001") == Price(1.0, 5.0, "builtin")
    # claude-opus-5 is a real, pricier model: a loose prefix must not price it
    assert builtin_price("claude-opus-5").input_per_mtok == 5.0
    assert builtin_price("claude-opus-5-5-preview") is None
    assert builtin_price("anthropic/claude-opus-5-5") is None
    assert builtin_price("qwen3:8b") is None


def test_price_math():
    assert Price(3.0, 15.0, "price").cost(2_000, 500) == pytest.approx(0.0135)
    assert Price(0.0, 0.0, "price").cost(10**6, 10**6) == 0.0


def test_user_price_overrides_builtin_and_server_cost_wins(conn):
    assert cost_of(conn, "claude-opus-5-5", _t()) == (24.0, "builtin")
    crud.set_price(conn, "claude-opus-5-5", 1.0, 2.0)
    assert cost_of(conn, "claude-opus-5-5", _t()) == (3.0, "price")
    # the endpoint billed it: trust that over any table
    assert cost_of(conn, "claude-opus-5-5", _t(reported=0.0123)) == (0.0123, "server")
    assert cost_of(conn, "claude-opus-5-5", _t(reported=0.0)) == (0.0, "server")


def test_unknown_model_has_unknown_cost(conn):
    assert resolve(conn, "llama3") is None
    assert cost_of(conn, "llama3", _t()) == (None, "")


def test_zero_price_means_free_not_unknown(conn):
    crud.set_price(conn, "llama3", 0, 0)
    assert cost_of(conn, "llama3", _t()) == (0.0, "price")


def _result(conn, model, tokens_in, tokens_out, status="PASS", cost=None):
    sid = conn.execute("SELECT id FROM servers LIMIT 1").fetchone()
    sid = sid[0] if sid else crud.add_server(conn, "s", "http://x/v1", None)
    run = crud.new_run(conn, sid, model, "en")
    return crud.save_result(
        conn,
        run,
        "p",
        "text",
        status,
        "",
        "",
        Telemetry(tokens_in=tokens_in, tokens_out=tokens_out),
        cost_usd=cost,
        cost_source="server" if cost is not None else "",
    )


def _cost(conn, rid):
    return conn.execute("SELECT cost_usd, cost_source FROM results WHERE id = ?", (rid,)).fetchone()


def test_set_price_fills_only_unpriced_results_of_that_model(conn):
    free = _result(conn, "deepseek-chat", 1_000_000, 0)
    paid = _result(conn, "deepseek-chat", 1_000_000, 0, cost=9.99)
    other = _result(conn, "other", 1_000_000, 0)
    pending = _result(conn, "deepseek-chat", 1_000_000, 0, status="PENDING")
    empty = _result(conn, "deepseek-chat", 0, 0, status="ERROR")

    assert crud.set_price(conn, "deepseek-chat", 0.5, 2.0) == 1
    assert tuple(_cost(conn, free)) == (0.5, "price")
    assert tuple(_cost(conn, paid)) == (9.99, "server")  # what it cost at the time
    assert tuple(_cost(conn, other)) == (None, "")
    assert tuple(_cost(conn, pending)) == (None, "")
    assert tuple(_cost(conn, empty)) == (None, "")  # never ran, nothing to charge

    # a new price does not rewrite history
    assert crud.set_price(conn, "deepseek-chat", 100.0, 100.0) == 0
    assert tuple(_cost(conn, free)) == (0.5, "price")
    assert crud.list_prices(conn)[0]["input_per_mtok"] == 100.0


def test_delete_price_keeps_recorded_costs(conn):
    rid = _result(conn, "m", 1_000_000, 0)
    crud.set_price(conn, "m", 1.0, 1.0)
    assert crud.delete_price(conn, "m") is True
    assert crud.delete_price(conn, "m") is False
    assert _cost(conn, rid)[0] == 1.0


def test_negative_price_is_rejected_by_the_schema(conn):
    with pytest.raises(sqlite3.IntegrityError):
        crud.set_price(conn, "m", -1.0, 1.0)


def test_leaderboard_sums_known_costs_only(conn):
    _result(conn, "m", 0, 0, cost=0.25)
    _result(conn, "m", 0, 0, cost=0.5)
    _result(conn, "m", 10, 10)  # unknown cost, not a zero
    [row] = crud.leaderboard(conn)
    assert row["total_cost"] == pytest.approx(0.75)
    assert row["priced"] == 2 and row["runs"] == 3


def test_history_carries_cost(conn):
    _result(conn, "m", 0, 0, cost=0.1)
    [row] = crud.history_filtered(conn)
    assert row["cost_usd"] == 0.1 and row["cost_source"] == "server"


def test_builtin_table_prices_are_sane():
    for model, (i, o) in pricing.BUILTIN.items():
        assert model.startswith("claude-")
        assert 0 < i < o, model


def test_migration_adds_cost_columns_and_prices_to_an_old_db(tmp_path):
    from llm_eval.db.connection import connect
    from llm_eval.db.schema import SCHEMA

    old = tmp_path / "old.db"
    raw = sqlite3.connect(old)
    stripped = SCHEMA.replace(
        "  cost_usd       REAL,                      -- NULL: no price known when it ran\n", ""
    ).replace("  cost_source    TEXT NOT NULL DEFAULT '',  -- server | price | builtin\n", "")
    stripped = (
        stripped[: stripped.index("-- USD per million")]
        + stripped[stripped.index("CREATE INDEX") :]
    )
    raw.executescript(stripped)
    raw.execute("INSERT INTO servers (name, base_url) VALUES ('s', 'http://x/v1')")
    raw.execute("INSERT INTO runs (server_id, model) VALUES (1, 'm')")
    raw.execute(
        "INSERT INTO results (run_id, prompt_id, eval_type, status, tokens_in) "
        "VALUES (1, 'p', 'text', 'PASS', 1000000)"
    )
    raw.commit()
    cols = {r[1] for r in raw.execute("PRAGMA table_info(results)")}
    assert "cost_usd" not in cols
    raw.close()

    c = connect(old)
    assert {"cost_usd", "cost_source"} <= {r[1] for r in c.execute("PRAGMA table_info(results)")}
    # old rows read as unknown, and a price fills them
    assert crud.set_price(c, "m", 2.0, 0) == 1
    assert c.execute("SELECT cost_usd FROM results").fetchone()[0] == 2.0
    c.close()
    connect(old).close()  # idempotent


# ---- engine ------------------------------------------------------------------


def _anthropic_server(conn):
    sid = crud.add_server(conn, "claude", "https://api.anthropic.com", "sk", "anthropic")
    return next(s for s in crud.list_servers(conn) if s.id == sid)


def _prompt(**over):
    from llm_eval.core.prompts import Prompt

    base = {
        "id": "t",
        "category": "logic",
        "eval_type": "text",
        "check": {"type": "final_number", "expected": 4},
        "locales": {"en": {"name": "T", "system_prompt": "be terse", "prompt": "2+2?"}},
    }
    base.update(over)
    return Prompt.model_validate(base)


def test_engine_anthropic_run_records_builtin_cost_and_no_temperature(conn, tmp_path):
    from llm_eval.core.client import GenerationParams
    from llm_eval.core.engine import evaluate
    from tests.fake_anthropic import FakeAnthropic, messages_sse, text_block

    server = _anthropic_server(conn)
    run = crud.new_run(conn, server.id, "claude-haiku-4-5", "en")
    fake = FakeAnthropic(
        [messages_sse([text_block("4")], usage_in=2_000, usage_out=500)],
        output_caps={"claude-haiku-4-5": 64_000},
    )
    out = evaluate(
        conn,
        run,
        server,
        "claude-haiku-4-5",
        _prompt(),
        "en",
        GenerationParams(temperature=0.7, timeout_s=5),
        http_client=fake.client(),
        traces_dir=tmp_path,
    )
    assert out.status == "PASS"
    # 2000 * $1/M + 500 * $5/M
    assert out.cost_usd == pytest.approx(0.0045) and out.cost_source == "builtin"
    row = crud.get_result(conn, out.result_id)
    assert row["cost_usd"] == pytest.approx(0.0045)
    assert row["temperature"] is None  # never sent, so never recorded
    assert "temperature" not in fake.body()


def test_engine_anthropic_refusal_is_labelled(conn, tmp_path):
    from llm_eval.core.client import GenerationParams
    from llm_eval.core.engine import evaluate
    from tests.fake_anthropic import FakeAnthropic, messages_sse

    server = _anthropic_server(conn)
    run = crud.new_run(conn, server.id, "claude-x", "en")
    fake = FakeAnthropic([messages_sse([], stop_reason="refusal")])
    out = evaluate(
        conn,
        run,
        server,
        "claude-x",
        _prompt(),
        "en",
        GenerationParams(timeout_s=5),
        http_client=fake.client(),
        traces_dir=tmp_path,
    )
    assert out.status == "FAIL" and "the model refused" in out.verdict.detail


def test_engine_anthropic_fault_uses_the_broken_key(conn, tmp_path):
    from llm_eval.core.client import GenerationParams
    from llm_eval.core.engine import evaluate
    from tests.fake_anthropic import FakeAnthropic, error

    server = _anthropic_server(conn)
    run = crud.new_run(conn, server.id, "claude-x", "en")
    fake = FakeAnthropic([error(401, "invalid x-api-key")], output_caps={"claude-x": 1000})
    p = _prompt(eval_type="error_test", check={"type": "none"}, fault={"kind": "bad_key"})
    out = evaluate(
        conn,
        run,
        server,
        "claude-x",
        p,
        "en",
        GenerationParams(timeout_s=5),
        http_client=fake.client(),
        traces_dir=tmp_path,
    )
    assert out.status == "PASS" and "authentication failed" in out.verdict.detail
    assert fake.requests[0].headers["x-api-key"] == "invalid-key-for-fault-injection"
    assert out.cost_usd is None  # nothing ran, nothing to charge
