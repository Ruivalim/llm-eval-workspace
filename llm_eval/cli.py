"""Headless battery: run every auto-checked prompt against one model and
print PASS/FAIL per prompt, exit 1 if anything failed. Same engine as the UI.

    python -m llm_eval.cli --model NAME [--locale en] [--server NAME] [--repeat N]
"""

from __future__ import annotations

import argparse
import sqlite3
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass

from llm_eval.config import load_settings
from llm_eval.core import engine
from llm_eval.core.client import EFFORTS, GenerationParams
from llm_eval.core.engine import Outcome
from llm_eval.core.prompts import Library, Prompt, load_library
from llm_eval.db import crud
from llm_eval.db.crud import Server
from llm_eval.db.init_db import init
from llm_eval.logging_setup import configure as configure_logging


@dataclass
class Summary:
    prompt_id: str
    statuses: list[str]
    ttft: list[int]
    tps: list[float]
    details: list[str]

    @property
    def ok(self) -> bool:
        return all(s == "PASS" for s in self.statuses)

    @property
    def status(self) -> str:
        if self.ok:
            return "PASS"
        return "FAIL" if "PASS" in self.statuses or "FAIL" in self.statuses else "ERROR"


def mean_sd(xs: list[float]) -> str:
    if not xs:
        return "n/a"
    if len(xs) == 1:
        return f"{xs[0]:.0f}"
    return f"{statistics.mean(xs):.0f} ± {statistics.stdev(xs):.0f}"


def pick_server(servers: list[Server], name: str | None) -> Server:
    if not servers:
        raise SystemExit("no servers configured; add one in the UI or seed the DB")
    if name is None:
        return servers[0]
    for s in servers:
        if s.name == name:
            return s
    raise SystemExit(f"unknown server {name!r}; known: {', '.join(s.name for s in servers)}")


def run_battery(
    conn: sqlite3.Connection,
    server: Server,
    model: str,
    lib: Library,
    locale: str,
    repeat: int,
    params: GenerationParams,
    evaluate: Callable[..., Outcome] | None = None,
    out=sys.stdout,
    traces_dir=None,
    allow_code_exec: bool = False,
) -> list[Summary]:
    evaluate = evaluate or engine.evaluate
    prompts: list[Prompt] = [p for p in lib.prompts if engine.is_auto(p)]
    run_id = crud.new_run(
        conn, server.id, model, locale, notes=f"battery x{repeat}", effort=params.effort
    )
    summaries: list[Summary] = []
    spent: float | None = None  # None until some result has a known cost
    for p in prompts:
        s = Summary(p.id, [], [], [], [])
        for _ in range(repeat):
            o = evaluate(
                conn,
                run_id,
                server,
                model,
                p,
                locale,
                params,
                traces_dir=traces_dir,
                allow_code_exec=allow_code_exec,
            )
            s.statuses.append(o.status)
            s.details.append(o.verdict.detail)
            if o.cost_usd is not None:
                spent = (spent or 0.0) + o.cost_usd
            if o.result:
                s.ttft.append(o.result.telemetry.ttft_ms)
                s.tps.append(o.result.telemetry.tps)
        summaries.append(s)
        print(
            f"{s.status:5} {p.id:32} ttft {mean_sd(s.ttft):>12} ms  "
            f"tps {mean_sd(s.tps):>10}  {s.details[-1]}",
            file=out,
        )
    passed = sum(s.ok for s in summaries)
    shown = model if params.effort == "default" else f"{model} (effort {params.effort})"
    print(f"\n{passed}/{len(summaries)} passed, model {shown}, locale {locale}", file=out)
    if spent is not None:
        print(f"cost ${spent:.4f} (results without a known price are not counted)", file=out)
    if traces_dir is not None:
        print(f"traces in {traces_dir}", file=out)
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm-eval battery")
    parser.add_argument("--model", required=True)
    parser.add_argument("--locale", default=None)
    parser.add_argument("--server", default=None, help="server name (default: first)")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument(
        "--effort",
        choices=EFFORTS,
        default="default",
        help="thinking effort; default sends nothing, off turns thinking off",
    )
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")
    if not args.model.strip():
        parser.error("--model must not be empty")
    settings = load_settings()
    configure_logging(settings.logs_dir)
    conn = init(settings)
    server = pick_server(crud.list_servers(conn), args.server)
    lib = load_library()
    locale = args.locale or settings.locale
    summaries = run_battery(
        conn,
        server,
        args.model,
        lib,
        locale,
        args.repeat,
        GenerationParams(timeout_s=args.timeout, effort=args.effort),
        traces_dir=settings.traces_dir,
        allow_code_exec=settings.allow_code_exec,
    )
    return 0 if all(s.ok for s in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
