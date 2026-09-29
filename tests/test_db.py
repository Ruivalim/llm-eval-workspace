import sqlite3

import pytest

from llm_eval.config import Settings
from llm_eval.core.telemetry import Telemetry
from llm_eval.db import crud
from llm_eval.db.init_db import seed_default_server


def _settings(**kw) -> Settings:
    return Settings(
        kw.pop("db_path", ":memory:"), kw.pop("base_url", ""), kw.pop("key", None), "en", **kw
    )


def test_first_run_shows_no_server_unless_the_env_asks_for_one(conn):
    """A stranger's first run must not open on somebody else's endpoint."""
    seed_default_server(conn, _settings())
    assert crud.list_servers(conn) == []

    seed_default_server(
        conn,
        _settings(base_url="http://localhost:11434/v1", key="k", default_flavor="unsloth"),
    )
    servers = crud.list_servers(conn)
    assert [(s.name, s.base_url, s.flavor) for s in servers] == [
        ("default", "http://localhost:11434/v1", "unsloth")
    ]
    # And it never overwrites what the user already configured.
    crud.add_server(conn, "mine", "http://mine/v1", None)
    seed_default_server(conn, _settings(base_url="http://other/v1"))
    assert len(crud.list_servers(conn)) == 2


def test_server_roundtrip_and_unique_name(conn):
    sid = crud.add_server(conn, "local", "http://localhost:11434/v1/", None)
    servers = crud.list_servers(conn)
    assert servers[0].id == sid and servers[0].base_url == "http://localhost:11434/v1"
    with pytest.raises(sqlite3.IntegrityError):
        crud.add_server(conn, "local", "http://x", None)


def test_result_status_is_constrained(conn):
    sid = crud.add_server(conn, "s", "http://x", None)
    run = crud.new_run(conn, sid, "m", "en")
    with pytest.raises(sqlite3.IntegrityError):
        crud.save_result(conn, run, "p", "text", "MAYBE", "", "", Telemetry())


def test_history_and_leaderboard(conn):
    sid = crud.add_server(conn, "s", "http://x", None)
    run = crud.new_run(conn, sid, "model-a", "en")
    t = Telemetry(ttft_ms=100, total_ms=1000, tokens_out=90, tps=100.0)
    r1 = crud.save_result(conn, run, "p1", "text", "PASS", "ok", "4", t)
    crud.save_result(conn, run, "p2", "text", "FAIL", "no", "5", t)
    crud.set_human_score(conn, r1, 8, "fine")
    crud.set_human_score(conn, r1, 9)  # upsert

    rows = crud.history(conn)
    assert [r["prompt_id"] for r in rows] == ["p2", "p1"]
    assert rows[1]["human_score"] == 9

    lb = crud.leaderboard(conn)
    assert lb[0]["model"] == "model-a"
    assert lb[0]["pass_rate"] == 50.0
    assert lb[0]["avg_human"] == 9.0


def test_leaderboard_counts_prompts_not_runs(conn):
    """Repeats don't inflate the fraction: passed and denominator are one row
    per distinct prompt, latest-run-wins."""
    sid = crud.add_server(conn, "s", "http://x", None)
    t = Telemetry(ttft_ms=100, total_ms=1000, tokens_out=90, tps=100.0)
    run = crud.new_run(conn, sid, "model-a", "en")
    crud.save_result(conn, run, "p1", "text", "PASS", "ok", "4", t)
    crud.save_result(conn, run, "p1", "text", "FAIL", "ok", "4", t)  # re-run, later id wins
    crud.save_result(conn, run, "p2", "text", "PASS", "ok", "4", t)

    lb = crud.leaderboard(conn)
    assert lb[0]["runs"] == 3  # total runs, repeats included
    assert lb[0]["prompts"] == 2  # distinct prompts, latest-run-wins
    assert lb[0]["passed"] == 1  # p1's latest run FAILs
    assert lb[0]["pass_rate"] == 50.0


def test_deleting_server_with_runs_is_blocked(conn):
    sid = crud.add_server(conn, "s", "http://x", None)
    crud.new_run(conn, sid, "m", "en")
    with pytest.raises(sqlite3.IntegrityError):
        crud.delete_server(conn, sid)


def test_run_records_parallel_requests(conn):
    """A batch at 4 in flight has to leave that number on the run: t/s and ttft from
    a loaded server are not comparable with a sequential run."""
    sid = crud.add_server(conn, "s", "http://x", None)
    seq = crud.new_run(conn, sid, "m", "en")
    par = crud.new_run(conn, sid, "m", "en", parallel=4)
    crud.save_result(conn, par, "p", "text", "PASS", "", "4", Telemetry())

    row = conn.execute("SELECT parallel FROM runs WHERE id = ?", (seq,)).fetchone()
    assert row["parallel"] == 1
    assert crud.history(conn)[0]["parallel"] == 4
    assert crud.history_filtered(conn)[0]["parallel"] == 4
    assert crud.get_result(conn, 1)["parallel"] == 4


def test_migration_adds_trace_path_and_parallel_to_an_old_db(tmp_path):
    import sqlite3 as _sq

    from llm_eval.db.connection import connect
    from llm_eval.db.schema import SCHEMA

    old = tmp_path / "old.db"
    raw = _sq.connect(old)
    stripped = SCHEMA.replace("  trace_path     TEXT NOT NULL DEFAULT '',\n", "").replace(
        "  parallel   INTEGER NOT NULL DEFAULT 1,  -- requests in flight at the time\n",
        "",
    )
    raw.executescript(stripped)
    raw.close()
    assert "trace_path" not in {
        r[1] for r in _sq.connect(old).execute("PRAGMA table_info(results)")
    }
    assert "parallel" not in {r[1] for r in _sq.connect(old).execute("PRAGMA table_info(runs)")}
    c = connect(old)
    assert "trace_path" in {r[1] for r in c.execute("PRAGMA table_info(results)")}
    assert "parallel" in {r[1] for r in c.execute("PRAGMA table_info(runs)")}
    c.close()
    connect(old).close()  # idempotent


def test_settings_upsert(conn):
    assert crud.get_setting(conn, "k", "dflt") == "dflt"
    crud.set_setting(conn, "k", "a")
    crud.set_setting(conn, "k", "b")
    assert crud.get_setting(conn, "k") == "b"


def test_delete_result_drops_human_score_and_orphan_run(conn):
    sid = crud.add_server(conn, "s", "http://x", None)
    run = crud.new_run(conn, sid, "m", "en")
    rid = crud.save_result(conn, run, "p", "text", "PASS", "", "4", Telemetry())
    crud.set_human_score(conn, rid, 7)

    row = crud.delete_result(conn, rid)
    assert row is not None and row["id"] == rid
    assert conn.execute("SELECT COUNT(*) FROM human_scores").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    assert crud.delete_result(conn, rid) is None


def test_clear_results_keeps_runs_that_still_have_results(conn):
    sid = crud.add_server(conn, "s", "http://x", None)
    run = crud.new_run(conn, sid, "m", "en")
    crud.save_result(conn, run, "keep", "text", "PASS", "", "4", Telemetry())
    crud.save_result(conn, run, "drop", "text", "FAIL", "", "3", Telemetry())

    count, rows = crud.clear_results(conn, status="FAIL")
    assert count == 1 and [r["prompt_id"] for r in rows] == ["drop"]
    assert [r["prompt_id"] for r in crud.history(conn)] == ["keep"]
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1

    count, rows = crud.clear_results(conn)
    assert count == 1 and [r["prompt_id"] for r in rows] == ["keep"]
    assert crud.history(conn) == []
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
