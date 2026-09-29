# Security

`llm-eval-workspace` is a local, single-user bench. It has no authentication and
no concept of more than one user, so what follows is less "policy" and more the
list of things that will bite you if you treat it as a service.

## Threat model

You run this on your own machine, against an LLM endpoint you configured, and you
look at the answers. The interesting risk is not someone attacking the bench from
outside: it is that the bench executes and renders text a language model wrote.

**Never expose the API to an untrusted network.** It holds your LLM API keys and
will call any `base_url` that is posted to it. Anyone who can reach it can spend
your tokens and drive requests to hosts of their choosing. `make api` binds
`127.0.0.1`; the container image listens on `0.0.0.0` because it is meant to be
published deliberately, behind a reverse proxy or a VPN, by someone who decided
that is fine.

## Hosted endpoints cost money

Against a hosted API every run is billed to your key. A batch sends up to 16
requests at a time, and an `error_test` prompt with `fault: {kind: huge_prompt}`
sends a couple hundred thousand tokens on purpose. The bench records what each
run cost when it knows a price, but it does not stop you from spending: set a
spending limit with the provider.

Keys go to the endpoint in a request header. A remote `base_url` over plain
`http://` sends that key in clear text across the network; use `https://` for
anything that is not on your machine or your VPN.

An `anthropic` server with no key configured sends an empty one and gets a 401.
It never falls back to an `ANTHROPIC_API_KEY` or login profile that happens to be
on the machine running the bench.

## What is stored, and where

- API keys are stored **unencrypted** in the SQLite file (`data/llm_eval.db`).
  `GET /api/servers` never returns them, but the file itself is readable by
  whoever can read it.
- Execution traces (`data/traces/`) record the request, every chunk the server
  sent, the model and the endpoint it ran against. They do not record the key.
  Sharing a trace shares your server's address.
- Everything lives under `data/` in the checkout, or under `/data` in the
  container. Nothing is sent anywhere except to the endpoints you configured.

## Running model output

- `render_html` answers are shown in an iframe with `sandbox="allow-scripts"` and
  no `allow-same-origin`, so the model's JavaScript runs in an opaque origin. The
  same document served from `/api/artifacts/{name}` carries
  `Content-Security-Policy: sandbox allow-scripts`, so opening it in a new tab
  does not hand it the API either.
- Markdown is sanitized with DOMPurify.
- `js_harness` checks execute JavaScript extracted from the answer with `node` or
  `bun`, **on your machine, with no sandbox**: filesystem and network are
  reachable from that code. It is therefore off by default and refuses to run
  unless you set `LLM_EVAL_ALLOW_CODE_EXEC=1`. The child process gets a minimal
  environment (`PATH` only) and Bun is told not to auto-load `.env`, so the model
  cannot read your API keys out of the process environment. Treat enabling it as
  running an untrusted script someone else wrote.

## Reporting

For anything sensitive, open a private security advisory on the repository
(GitHub: Security, then Report a vulnerability). For everything else, a regular
issue is fine.
