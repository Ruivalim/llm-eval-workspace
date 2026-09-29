"""Isolated query functions. Parameters always bound, never formatted in."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from llm_eval.core.telemetry import Telemetry

FLAVORS = ("openai", "unsloth", "anthropic")


@dataclass(frozen=True)
class Server:
    id: int
    name: str
    flavor: str
    base_url: str
    api_key: str | None


# ---- servers ---------------------------------------------------------------


def add_server(
    conn: sqlite3.Connection, name: str, base_url: str, api_key: str | None, flavor: str = "openai"
) -> int:
    cur = conn.execute(
        "INSERT INTO servers (name, flavor, base_url, api_key) VALUES (?, ?, ?, ?)",
        (name, flavor, base_url.rstrip("/"), api_key),
    )
    conn.commit()
    return int(cur.lastrowid)


def list_servers(conn: sqlite3.Connection) -> list[Server]:
    rows = conn.execute("SELECT * FROM servers ORDER BY id").fetchall()
    return [Server(r["id"], r["name"], r["flavor"], r["base_url"], r["api_key"]) for r in rows]


def update_server(
    conn: sqlite3.Connection,
    server_id: int,
    name: str,
    base_url: str,
    api_key: str | None,
    flavor: str,
) -> None:
    conn.execute(
        "UPDATE servers SET name = ?, base_url = ?, api_key = ?, flavor = ? WHERE id = ?",
        (name, base_url.rstrip("/"), api_key, flavor, server_id),
    )
    conn.commit()


def delete_server(conn: sqlite3.Connection, server_id: int) -> None:
    conn.execute("DELETE FROM servers WHERE id = ?", (server_id,))
    conn.commit()


# ---- runs / results --------------------------------------------------------


def new_run(
    conn: sqlite3.Connection,
    server_id: int,
    model: str,
    locale: str,
    notes: str = "",
    parallel: int = 1,
    effort: str = "default",
) -> int:
    """One run row = one evaluation. `parallel` records how many requests the batch
    had in flight, because tps/ttft from a shared model server are not comparable
    across different concurrency levels. `effort` is recorded for the same
    reason: one model at two effort levels is two things to compare."""
    cur = conn.execute(
        """INSERT INTO runs (server_id, model, locale, parallel, notes, effort)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (server_id, model, locale, parallel, notes, effort),
    )
    conn.commit()
    return int(cur.lastrowid)


def save_result(
    conn: sqlite3.Connection,
    run_id: int,
    prompt_id: str,
    eval_type: str,
    status: str,
    detail: str,
    answer: str,
    telemetry: Telemetry,
    *,
    reasoning: str = "",
    artifact_path: str = "",
    trace_path: str = "",
    temperature: float | None = None,
    max_tokens: int | None = None,
    cost_usd: float | None = None,
    cost_source: str = "",
) -> int:
    cur = conn.execute(
        """INSERT INTO results (run_id, prompt_id, eval_type, status, detail, answer, reasoning,
             artifact_path, trace_path, temperature, max_tokens, ttft_ms, total_ms, tokens_in,
             tokens_out, tokens_think, tps, cost_usd, cost_source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            prompt_id,
            eval_type,
            status,
            detail,
            answer,
            reasoning,
            artifact_path,
            trace_path,
            temperature,
            max_tokens,
            telemetry.ttft_ms,
            telemetry.total_ms,
            telemetry.tokens_in,
            telemetry.tokens_out,
            telemetry.tokens_think,
            telemetry.tps,
            cost_usd,
            cost_source,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def set_human_score(
    conn: sqlite3.Connection, result_id: int, score: int, comment: str = ""
) -> None:
    conn.execute(
        """INSERT INTO human_scores (result_id, score, comment) VALUES (?, ?, ?)
           ON CONFLICT(result_id) DO UPDATE
             SET score = excluded.score, comment = excluded.comment""",
        (result_id, score, comment),
    )
    conn.commit()


def history(conn: sqlite3.Connection, limit: int = 200) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT r.*, run.model, run.locale, run.server_id, run.parallel, run.effort,
                  s.name AS server_name, h.score AS human_score
           FROM results r
           JOIN runs run ON run.id = r.run_id
           JOIN servers s ON s.id = run.server_id
           LEFT JOIN human_scores h ON h.result_id = r.id
           ORDER BY r.id DESC LIMIT ?""",
        (limit,),
    ).fetchall()


def leaderboard(
    conn: sqlite3.Connection,
    *,
    model: str | None = None,
    prompt_id: str | None = None,
    locale: str | None = None,
) -> list[sqlite3.Row]:
    """Comparable automatic results, plus human scores from every saved result.

    Passed and the denominator count one row per distinct prompt, using the
    latest run for that prompt (latest-run-wins). Repeats don't inflate either
    number, so the fraction is passes over prompts that exist, not total runs.
    """
    clause, args = _result_filters(model, prompt_id, None, locale)
    return conn.execute(
        f"""WITH latest AS (
               SELECT r.id, r.prompt_id, r.status,
                      run.id AS run_id, run.model, run.effort, run.locale, run.server_id
               FROM results r
               JOIN runs run ON run.id = r.run_id
               {clause}
               AND r.status <> 'PENDING'
               AND r.id = (
                 SELECT MAX(r2.id)
                 FROM results r2
                 JOIN runs run2 ON run2.id = r2.run_id
                 WHERE r2.prompt_id = r.prompt_id
                   AND run2.model = run.model
                   AND run2.effort = run.effort
                   AND run2.locale = run.locale
                   AND run2.server_id = run.server_id
               )
             ),
             run_stats AS (
               SELECT run.model,
                      run.effort,
                      COUNT(*) AS runs,
                      ROUND(AVG(r.ttft_ms)) AS avg_ttft_ms,
                      ROUND(AVG(r.tps), 1) AS avg_tps,
                      SUM(r.cost_usd) AS total_cost,
                      COUNT(r.cost_usd) AS priced
               FROM results r
               JOIN runs run ON run.id = r.run_id
               {clause}
               AND r.status <> 'PENDING'
               GROUP BY run.model, run.effort
             )
            SELECT run_stats.model,
                   run_stats.effort,
                   run_stats.runs,
                   SUM(latest.status = 'PASS') AS passed,
                   COUNT(DISTINCT latest.prompt_id) AS prompts,
                   ROUND(100.0 * SUM(latest.status = 'PASS') /
                         COUNT(DISTINCT latest.prompt_id), 1) AS pass_rate,
                   run_stats.avg_ttft_ms,
                   run_stats.avg_tps,
                   run_stats.total_cost,
                   run_stats.priced,
                   (SELECT ROUND(AVG(h2.score), 1)
                    FROM results r2
                    JOIN runs run2 ON run2.id = r2.run_id
                    JOIN human_scores h2 ON h2.result_id = r2.id
                    WHERE run2.model = run_stats.model
                      AND run2.effort = run_stats.effort
                      AND (? IS NULL OR r2.prompt_id = ?)
                      AND (? IS NULL OR run2.locale = ?)) AS avg_human
            FROM run_stats
            LEFT JOIN latest
              ON latest.model = run_stats.model AND latest.effort = run_stats.effort
            GROUP BY run_stats.model, run_stats.effort
            ORDER BY pass_rate DESC, avg_tps DESC""",
        [*args, prompt_id, prompt_id, locale, locale],
    ).fetchall()


def get_result(conn: sqlite3.Connection, result_id: int) -> sqlite3.Row | None:
    return conn.execute(
        """SELECT r.*, run.model, run.locale, run.server_id, run.parallel, run.effort,
                  s.name AS server_name, h.score AS human_score, h.comment AS human_comment
           FROM results r
           JOIN runs run ON run.id = r.run_id
           JOIN servers s ON s.id = run.server_id
           LEFT JOIN human_scores h ON h.result_id = r.id
           WHERE r.id = ?""",
        (result_id,),
    ).fetchone()


def history_filtered(
    conn: sqlite3.Connection,
    *,
    model: str | None = None,
    prompt_id: str | None = None,
    status: str | None = None,
    locale: str | None = None,
    limit: int = 200,
) -> list[sqlite3.Row]:
    clause, args = _result_filters(model, prompt_id, status, locale)
    return conn.execute(
        f"""SELECT r.id, r.run_id, r.prompt_id, r.eval_type, r.status, r.detail, r.artifact_path,
                   r.trace_path,
                   r.temperature, r.max_tokens, r.ttft_ms, r.total_ms, r.tokens_in, r.tokens_out,
                   r.tokens_think, r.tps, r.cost_usd, r.cost_source, r.created_at,
                   run.model, run.locale, run.server_id, run.parallel, run.effort,
                   s.name AS server_name, h.score AS human_score
            FROM results r
            JOIN runs run ON run.id = r.run_id
            JOIN servers s ON s.id = run.server_id
            LEFT JOIN human_scores h ON h.result_id = r.id
            {clause}
            ORDER BY r.id DESC LIMIT ?""",
        (*args, limit),
    ).fetchall()


def _result_filters(
    model: str | None, prompt_id: str | None, status: str | None, locale: str | None
) -> tuple[str, list[str]]:
    where, args = [], []
    for col, val in (
        ("run.model", model),
        ("r.prompt_id", prompt_id),
        ("r.status", status),
        ("run.locale", locale),
    ):
        if val:
            where.append(f"{col} = ?")
            args.append(val)
    return ("WHERE " + " AND ".join(where)) if where else "", args


def _orphan_runs(conn: sqlite3.Connection) -> None:
    """A run exists only to group its results; drop the ones left empty."""
    conn.execute(
        "DELETE FROM runs WHERE NOT EXISTS (SELECT 1 FROM results WHERE results.run_id = runs.id)"
    )


def delete_result(conn: sqlite3.Connection, result_id: int) -> sqlite3.Row | None:
    """Delete one result (human score cascades) and return the row, or None.
    The caller owns the files: artifact_path and trace_path are returned for it."""
    row = conn.execute("SELECT * FROM results WHERE id = ?", (result_id,)).fetchone()
    if row is None:
        return None
    conn.execute("DELETE FROM results WHERE id = ?", (result_id,))
    _orphan_runs(conn)
    conn.commit()
    return row


def clear_results(
    conn: sqlite3.Connection,
    *,
    model: str | None = None,
    prompt_id: str | None = None,
    status: str | None = None,
    locale: str | None = None,
) -> tuple[int, list[sqlite3.Row]]:
    """Delete every result matching the same filters the history view uses.
    Returns (count, rows) so the caller can remove the artifact and trace files."""
    clause, args = _result_filters(model, prompt_id, status, locale)
    rows = conn.execute(
        f"""SELECT r.* FROM results r JOIN runs run ON run.id = r.run_id {clause}""", args
    ).fetchall()
    if not rows:
        return 0, []
    conn.execute(
        f"""DELETE FROM results WHERE id IN (
              SELECT r.id FROM results r JOIN runs run ON run.id = r.run_id {clause})""",
        args,
    )
    _orphan_runs(conn)
    conn.commit()
    return len(rows), rows


def matrix(
    conn: sqlite3.Connection,
    *,
    model: str | None = None,
    prompt_id: str | None = None,
    locale: str | None = None,
) -> list[sqlite3.Row]:
    """Latest result per model, server, locale and prompt."""
    clause, args = _result_filters(model, prompt_id, None, locale)
    return conn.execute(
        f"""SELECT r.id, r.prompt_id, r.status, r.tps, r.ttft_ms, r.detail,
                   run.model, run.effort, run.locale, run.server_id, h.score AS human_score
            FROM results r
            JOIN runs run ON run.id = r.run_id
            LEFT JOIN human_scores h ON h.result_id = r.id
            {clause}
            {"WHERE" if not clause else "AND"} r.id = (
                SELECT MAX(r2.id)
                FROM results r2 JOIN runs run2 ON run2.id = r2.run_id
                WHERE r2.prompt_id = r.prompt_id
                  AND run2.model = run.model
                  AND run2.effort = run.effort
                  AND run2.locale = run.locale
                  AND run2.server_id = run.server_id
            )
            ORDER BY run.model, run.effort, run.server_id, run.locale, r.prompt_id""",
        args,
    ).fetchall()


# ---- prices --------------------------------------------------------------------


def list_prices(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT model, input_per_mtok, output_per_mtok, updated_at FROM model_prices ORDER BY model"
    ).fetchall()


def set_price(
    conn: sqlite3.Connection, model: str, input_per_mtok: float, output_per_mtok: float
) -> int:
    """Upsert the price, then fill the cost of this model's results that ran
    without one. Results that already have a cost keep it: that is what they
    cost at the time. Returns how many results were filled."""
    conn.execute(
        """INSERT INTO model_prices (model, input_per_mtok, output_per_mtok) VALUES (?, ?, ?)
           ON CONFLICT(model) DO UPDATE
             SET input_per_mtok = excluded.input_per_mtok,
                 output_per_mtok = excluded.output_per_mtok,
                 updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')""",
        (model, input_per_mtok, output_per_mtok),
    )
    cur = conn.execute(
        """UPDATE results
           SET cost_usd = (tokens_in * ? + tokens_out * ?) / 1000000.0, cost_source = 'price'
           WHERE cost_usd IS NULL
             AND status <> 'PENDING'
             AND (tokens_in > 0 OR tokens_out > 0)
             AND run_id IN (SELECT id FROM runs WHERE model = ?)""",
        (input_per_mtok, output_per_mtok, model),
    )
    conn.commit()
    return cur.rowcount


def delete_price(conn: sqlite3.Connection, model: str) -> bool:
    """Forget the price. Costs already recorded stay."""
    cur = conn.execute("DELETE FROM model_prices WHERE model = ?", (model,))
    conn.commit()
    return cur.rowcount > 0


# ---- settings ----------------------------------------------------------------

GLOBAL_SYSTEM_PROMPT = "global_system_prompt"


def get_setting(conn: sqlite3.Connection, key: str, default: str = "") -> str:
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """INSERT INTO settings (key, value) VALUES (?, ?)
           ON CONFLICT(key) DO UPDATE
             SET value = excluded.value, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')""",
        (key, value),
    )
    conn.commit()
