# Prompt library format

Prompts live in `data/prompts/*.yaml`, one file per category, each file a list.
They are versioned with the code; the database only stores runs and results and
references prompts by `id`.

```yaml
- id: sally-daughters            # stable, kebab-case, unique across all files
  category: logic                # must exist in data/categories.yaml
  eval_type: text                # text | render_html | tool_call | error_test
  max_tokens: 4000               # optional; omitted = server default (recommended),
                                 # or the model's own output cap on anthropic servers
  tags: [reasoning]              # optional
  tools: [get_weather]           # tool_call only: mock tools offered to the model
  fault: {kind: timeout, seconds: 2}   # error_test only
  check:                         # automatic verdict, see table below
    type: final_number
    expected: 4
  locales:
    en:                          # mandatory, the default and the fallback
      name: Sally's daughters
      system_prompt: ""
      prompt: |
        ...
    pt-BR:                       # optional, any locale in SUPPORTED_LOCALES
      name: Filhas de Sally
      system_prompt: ""
      prompt: |
        ...
```

Rule: every locale of a prompt must demand the same output format, so the
same `check` works regardless of language. Format lines that the checker parses
(such as `Final: <number> apples, <location>`) stay in English in every locale.

## Check types (`llm_eval/core/checks.py`)

| type            | params                                              | passes when |
|-----------------|-----------------------------------------------------|-------------|
| `none`          |                                                     | always (needs human score) |
| `final_number`  | `expected`                                          | last number on the last content line equals expected |
| `regex`         | `pattern`                                           | pattern matches anywhere |
| `contains_all`  | `values`, `case_sensitive` (false)                  | every value present |
| `exact_lines`   | `count`, `prefix`                                   | exactly N non-empty lines, each with prefix |
| `json_equals`   | `expected` (object)                                 | answer parses as JSON and every key matches |
| `html_complete` | `min_chars`, `must_contain`                         | full `<html>…</html>`, long enough, has tags |
| `tool_called`   | `tools`, `final_contains`                           | every tool was called, final text has needles |
| `js_harness`    | `harness`, `must_match`, `must_not_contain`         | extracted JS + harness runs without throwing (bun or node), off by default |

Where a checker looks for a value, digit grouping does not matter: `55,212`,
`55.212` and `55212` are the same number for `final_number` and for the
`final_contains` needles of `tool_called`. Markdown fence lines (` ``` `) are
skipped when `final_number` looks for the last line, so an answer that wraps the
result in a code block still passes.

A checker that cannot run at all returns `Verdict(..., infra=True)`, and the run
is saved as `ERROR` instead of `FAIL`: a missing JS runtime is the machine's
problem, not the model's. Every check error message carries the failure, not the
dumped source around it.

`js_harness` executes JavaScript the model wrote, so it only runs when
the operator sets `LLM_EVAL_ALLOW_CODE_EXEC=1`; without it the check returns an
explained `ERROR`. Treat a prompt that uses it as running an untrusted script:
the child process gets no environment beyond `PATH`, but there is no sandbox
around the filesystem or the network. See `SECURITY.md`.

Add a checker: write `check_<name>(inp, params) -> Verdict` and register it in
`CHECKERS`. `tests/test_prompts.py` fails if a YAML references an unknown type.
`tests/test_checks.py` runs a known-correct implementation through every shipped
`js_harness`, so a wrong expectation in the YAML fails the suite.
