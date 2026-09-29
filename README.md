# llm-eval-workspace

A workbench for qualitative LLM evaluation, in the spirit of the fixed prompt
battery Sam Witteveen runs on every new model: same prompts, every model, watch
it stream, measure it, judge it.

Point it at a model (local or hosted), run a prompt or the whole battery, and it
keeps every answer with its telemetry, an automatic verdict, what it cost, and a
full trace of the request.

- **Any OpenAI-compatible endpoint**: Ollama, llama.cpp, vLLM, Unsloth Studio,
  or hosted APIs like DeepSeek, OpenRouter, OpenAI and Xiaomi MiMo.
- **Claude** through the Anthropic API (official SDK), plus gateways that speak
  the same API.
- **53 prompts** in 6 categories (logic, instruction following, coding, UI with
  live HTML rendering, function calling with mocked tools, resilience), each with
  an automatic pass/fail check, in English and Brazilian Portuguese.
- **Telemetry**: time to first token, tokens per second, token counts, thinking
  tokens.
- **Cost per run** in USD, from the endpoint's own billing, your per-model price,
  or Anthropic's list price.
- **Thinking effort** per run: model default, off, or a level.
- **History in SQLite**: leaderboard per model, a prompt by model matrix, human
  scores, and a downloadable JSONL trace of every raw chunk.

Desktop-only and single user: a bench for one person at a workstation, not a
service. No authentication, no multi-tenancy.

## Quickstart with Docker

```bash
docker compose up -d --build
```

Open http://127.0.0.1:8600, go to **Settings**, add a server (see
[Servers](#servers)), pick a model in the header, pick a prompt and press
**Run**.

State (database, traces, artifacts, logs) lives in the `llm-eval-data` Docker
volume, so it survives rebuilds. The port is bound to `127.0.0.1` on purpose;
read [Security](#security) before changing that. `PORT=8700 docker compose up -d`
picks another port.

## Running from source

Needs Python 3.12+ with [uv](https://docs.astral.sh/uv/), and
[Bun](https://bun.sh/).

```bash
make setup   # venv + Python deps, and bun install
make dev     # API on 127.0.0.1:8600 and Vite with hot reload on :5173
```

Open http://localhost:5173. `make build && make api` serves the compiled UI from
the API on http://127.0.0.1:8600 instead. From source the database is
`data/llm_eval.db`, separate from the Docker volume.

`make help` lists every target.

## Servers

Add them in **Settings**. The key is stored in the local database and only sent
to that server.

| Provider | flavor | base URL |
|---|---|---|
| Ollama | `openai` | `http://localhost:11434/v1` |
| llama.cpp | `openai` | `http://127.0.0.1:8080/v1` |
| vLLM | `openai` | `http://localhost:8000/v1` |
| DeepSeek | `openai` | `https://api.deepseek.com/v1` |
| OpenRouter | `openai` | `https://openrouter.ai/api/v1` |
| OpenAI | `openai` | `https://api.openai.com/v1` |
| Xiaomi MiMo | `openai` | `https://api.xiaomimimo.com/v1` |
| Xiaomi MiMo | `anthropic` | `https://api.xiaomimimo.com/anthropic` |
| Anthropic (Claude) | `anthropic` | `https://api.anthropic.com` |
| Unsloth Studio | `unsloth` | its `/v1` URL |

- **`openai`** is any OpenAI-compatible `/v1`. Rate limits (429) and busy servers
  (5xx) are retried twice, honoring `Retry-After` up to 30 s; each retry is in the
  trace, since the wait counts toward time to first token.
- **`anthropic`** goes through the official Anthropic SDK. Temperature and top_p
  are never sent (current Claude models reject them). A prompt with no
  `max_tokens` gets the model's own output cap, up to 64k. Gateways that serve
  this API under a path, like MiMo's `/anthropic`, work too: the key is also sent
  as a Bearer token, and models are listed from the gateway's root `/v1/models`
  when its Anthropic side has no model list.
- **`unsloth`** adds Unsloth Studio's loaded-model panel and lets the bench
  swap the loaded model.

## Using it

- **Bench**: pick a prompt in the sidebar and press Run. The answer streams, the
  verdict and telemetry land on the right, `render_html` answers render live in a
  sandboxed frame. temp, top-p, effort and timeout sit next to the Run button.
- **Batch**: "Run all auto-checked" plans the whole battery, skipping prompts the
  model already settled at the same effort, and can run up to 16 at once.
  Requests in flight are recorded with each result, because tokens per second
  under concurrency is not comparable with a sequential run.
- **Results**: pass rate per model, the prompt by model matrix, and the full
  history with filters. Each result opens with its answer, reasoning, trace and a
  0 to 10 human score.
- **Headless**: `make battery MODEL=<id> LOCALE=en EFFORT=default` runs every
  auto-checked prompt against the first server and prints a table.

### Thinking effort

| effort | `openai` flavor sends | `anthropic` flavor sends |
|---|---|---|
| `default` | nothing, the model runs as it ships | nothing |
| `off` | `thinking: {type: disabled}` (MiMo, DeepSeek, GLM, Kimi) | `thinking: {type: disabled}` |
| `low` to `max` | `reasoning_effort` | `output_config.effort` |

A model that does not take the setting answers with an error, shown as the run's
error. Some accept it and ignore it: MiMo only switches thinking on or off. The
leaderboard and the matrix keep each effort apart, so one model at `high` and at
`off` shows up as two rows.

### Cost

Every result stores what it cost in USD, from the first source that knows:

1. the endpoint, when it bills per request (OpenRouter's `usage.cost`);
2. your price for that model, set in **Settings > Prices** (USD per million
   input and output tokens) for the models of any server;
3. Anthropic's list price, built in for Claude model ids.

No price means unknown, shown as `-`, never as free. Saving a price fills in
earlier results of that model that ran without one; results that already have a
cost keep it. Cache discounts are not modelled.

A hosted endpoint spends money on every run, and a batch can send 16 at once. Set
a spending limit with the provider.

## Configuration

All optional. From source they go in `.env` (copy `.env.example`); Docker reads
the same file.

| Variable | What it does |
|---|---|
| `LLM_EVAL_DEFAULT_BASE_URL`, `LLM_EVAL_DEFAULT_API_KEY`, `LLM_EVAL_DEFAULT_FLAVOR` | Seed one server into a new database |
| `LLM_EVAL_ALLOW_CODE_EXEC=1` | Turn on `js_harness` checks, which run model-written JavaScript on your machine |
| `LLM_EVAL_LOCALE` | Default prompt language (`en` or `pt-BR`) |
| `LLM_EVAL_DB`, `LLM_EVAL_TRACES`, `LLM_EVAL_ARTIFACTS`, `LLM_EVAL_LOGS` | Where state lives (default: under `data/`) |
| `LLM_EVAL_LOG_LEVEL` | Application log level (default `INFO`) |

## Security

Read [SECURITY.md](SECURITY.md) before exposing this to anything.

- **No authentication, and API keys are stored in plain text** in the local
  SQLite file. Anyone who reaches the API can spend your keys and point the bench
  at any URL. Everything binds `127.0.0.1` by default; keep it on loopback or
  behind a VPN, never on the open internet.
- **It runs model output.** HTML answers render in a sandboxed iframe and are
  served with a CSP sandbox. `js_harness` checks execute JavaScript on your
  machine with no filesystem or network sandbox, so they are off unless
  `LLM_EVAL_ALLOW_CODE_EXEC=1`.
- A trace records the endpoint and model it ran against, never the key. Do not
  share one you would not share your server's address with.

## Writing prompts

Prompts are YAML files in `data/prompts/`, one per category, with the text per
locale and an automatic check. [docs/PROMPTS.md](docs/PROMPTS.md) covers the
format and every check type.

## Development

```bash
make check   # ruff, oxlint, tsc, pytest and vitest
make fmt     # format Python and TypeScript
make test    # tests only
```

Code is Python (FastAPI) in `llm_eval/` and React + TypeScript in `web/src/`.
New code comes with tests; the model side is faked with scripted SSE servers
(`tests/fake_server.py`, `tests/fake_anthropic.py`), so the suite needs no
network or key.

```
llm_eval/
  api.py             FastAPI app: servers, prices, SSE evaluate, history, artifacts
  cli.py             headless battery, same engine as the UI
  config.py          env and paths, read once
  core/
    engine.py        one evaluation end to end: messages, stream, check, save
    client.py        OpenAI-compatible streaming, tool loop, retries, effort
    anthropic_client.py  same contract over the Anthropic Messages API
    pricing.py       cost of a run, built-in Claude prices
    checks.py        automatic verdicts
    ...              prompts, telemetry, tool mock, trace, unsloth, html
  db/                SQLite schema, migrations, queries
web/src/             Bench, Results, Settings
data/prompts/        the prompt library
tests/               pytest (tests/) and vitest (web/src/**/*.test.ts*)
```

`AGENTS.md` has the notes for coding agents. `AGENTS.md`, `.claude/`,
`.mcp.json` and `opencode.json` also wire the repo to
[graft](https://www.npmjs.com/package/@nanonets/graft), a code graph for coding
agents; without graft installed they do nothing.

## License

MIT. See [LICENSE](LICENSE).
