# AGENTS.md

Notes for coding agents working on llm-eval-workspace. The README is the user
guide; this file is what you need to change the code without breaking it.

## What this is

A single-user bench that runs a fixed prompt battery against LLM endpoints
(OpenAI-compatible or the Anthropic Messages API), streams the answer, checks it
automatically, and stores every run with telemetry, cost and a raw trace.
FastAPI + SQLite in `llm_eval/`, React + TypeScript in `web/src/`.

## Commands

```bash
make setup    # venv + deps, bun install
make dev      # API :8600 + Vite :5173
make check    # ruff, oxlint, tsc, pytest, vitest
make fmt      # format everything
make build    # web/dist, served by the API
docker compose up -d --build   # the production image on 127.0.0.1:8600
```

Run `make check` before saying a change is done. For anything that touches a
provider, also run it for real at least once against a server (`make dev`, or
the API with a throwaway `LLM_EVAL_DB`).

## How a run flows

`api.evaluate` (SSE) and `cli.run_battery` both call `engine.evaluate`, so UI and
CLI numbers agree. `engine` builds the messages, picks the client by server
flavor in `_open_stream`, streams `Delta`s until one `ChatResult`, runs the check
(`core/checks.py`), prices it (`core/pricing.py`), saves the row and closes the
trace.

- `core/client.py`: OpenAI-compatible streaming, tool loop over `tool_mock`,
  repetition guard, retries (`open_with_retry`), effort mapping
  (`openai_effort`), fault injection (`apply_fault`).
- `core/anthropic_client.py`: the same contract over the Anthropic SDK
  (`stream_messages`), with `anthropic_effort` and gateway handling.
- `db/crud.py`: every query. `db/schema.py`: tables plus `MIGRATIONS`.

## Where changes go

- **New provider flavor**: add it to `crud.FLAVORS`, branch in
  `engine._open_stream` and in `api.models`, add the option in
  `web/src/components/Settings.tsx`, and test it with a scripted fake like
  `tests/fake_server.py` / `tests/fake_anthropic.py`.
- **New DB column**: put it in `SCHEMA` and append `(table, column, definition)`
  to `MIGRATIONS`, so existing databases gain it on start. Test the migration
  from a schema without the column.
- **New request parameter**: carry it in `GenerationParams` and rebuild params
  with `dataclasses.replace`, never positionally, or a field gets dropped.
- **New prompt or check**: `data/prompts/*.yaml`, format in `docs/PROMPTS.md`;
  checks in `core/checks.py`.

## Invariants that tests guard

- **Stopped is not a result.** A user stop raises `Stopped`; nothing is saved.
- **Unknown cost is `None`, never 0.** `set_price` fills results without a cost
  and never rewrites one already recorded. An endpoint-reported cost wins over
  any price.
- **Anthropic**: no temperature or top_p in the request (current models 400 on
  them). A server without a key sends `"none"`, never falling back to
  `ANTHROPIC_API_KEY` or a login profile on the machine. The whole assistant turn,
  thinking blocks included, goes back unchanged in the tool loop.
- **Effort `default` sends nothing.** Runs carry their effort, and leaderboard,
  matrix and batch planning (`planRows`) keep efforts apart.
- **Retries** happen only before the stream opens, only for 429/5xx, and a stop
  interrupts the wait.
- **Security**: the SPA catch-all serves only files inside `web/dist`
  (`spa_target`); artifacts go out with `Content-Security-Policy: sandbox`;
  `js_harness` runs only with `LLM_EVAL_ALLOW_CODE_EXEC=1`, with a minimal env.
  The API binds `127.0.0.1` unless told otherwise.

## Conventions

- Tests come with the code and need no network or key; `tests/conftest.py`
  replaces the retry backoff sleep. Cover what breaks (bad input, errors,
  concurrency, limits), not only the happy path.
- A failing test means the code is wrong until shown otherwise. If the test was
  wrong, say why instead of quietly changing it.
- Python and TypeScript code, their comments, README and UI text are in English;
  Makefile, Dockerfile and compose comments are in Portuguese. Comments explain
  why, not what.
- Never commit `.env`, `data/*.db`, traces, artifacts or logs (all ignored).

<!-- graft:start -->
## Graft — repo context graph

This repo is indexed in `graft/`: small linked markdown nodes that explain each
system and carry exact file:line spans, kept in sync with the code through git.

For ANY task here — understanding how something works, finding where code lives,
or scoping a change — get context from the graph before grepping or opening
source files. Re-ask freely (it's cheap) and reuse literal identifiers you
already have (symbol, error string, file name) as the query. New to this repo?
Run `graft map` first — a token-budgeted orientation (dir clusters, hubs,
hotspots), no LLM, no key.

- Run `graft ask "<your question>" --source` → ranked nodes with the relevant
  code spans inlined (each hit's ≤8-line crux by default; `--full` for whole
  definitions when the crux isn't enough). Match the tool to the task shape:
  for understanding or editing, the top node IS the answer — cite its
  `covers:` file:line spans and edit straight from `--source`. For
  exhaustive tasks ("every occurrence / every caller of this pattern"), ranked
  results are top-N, not complete — run `graft grep "<literal>"` instead
  (exhaustive over indexed files, grouped by enclosing symbol), falling back
  to raw `grep -rn` only for unindexed files.
- `graft skeleton <file>` → every definition's signature + span, ~10× cheaper
  than reading the file; use it to skim an API surface.
- `graft callers <symbol>` gives precomputed, exact edges — who calls this.
  Add `--direction out` for what it calls, or `--depth N` to walk
  transitively for the full blast radius. For structural questions, skip
  ranking and use this directly.
- Or browse: `graft/INDEX.md` lists every node; follow the links.
- Monorepos and folders of multiple repos rank fairly across sub-projects —
  hits carry `[scope/]` labels naming which one they're from. Narrow with
  `graft ask "<task>" --in <scope>/` once you know where you're working.

If a returned span is truncated ("+N more lines"), open the file at that exact
range before finalizing. Only open source files when a node genuinely lacks a
needed detail, and then at the exact file:line the node points to — never
re-read whole files.

After big code changes, refresh the graph with `graft build` (deterministic,
no API key, $0).
<!-- graft:end -->
