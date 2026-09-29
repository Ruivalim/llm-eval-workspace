"""SQLite schema. Prompts are versioned files in data/prompts; the DB keeps
servers, runs and results. Prompts are referenced by their stable string id."""

SCHEMA = """
CREATE TABLE IF NOT EXISTS servers (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  name       TEXT NOT NULL UNIQUE,
  flavor     TEXT NOT NULL DEFAULT 'openai',   -- openai | unsloth (adds load-model) | anthropic
  base_url   TEXT NOT NULL,
  api_key    TEXT,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS runs (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  server_id  INTEGER NOT NULL REFERENCES servers(id),
  model      TEXT NOT NULL,
  locale     TEXT NOT NULL DEFAULT 'en',
  parallel   INTEGER NOT NULL DEFAULT 1,  -- requests in flight at the time
  effort     TEXT NOT NULL DEFAULT 'default',  -- thinking effort asked for (client.EFFORTS)
  notes      TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS results (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id         INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  prompt_id      TEXT NOT NULL,
  eval_type      TEXT NOT NULL,
  status         TEXT NOT NULL CHECK (status IN ('PASS','FAIL','PARTIAL','ERROR','PENDING')),
  detail         TEXT NOT NULL DEFAULT '',
  answer         TEXT NOT NULL DEFAULT '',
  reasoning      TEXT NOT NULL DEFAULT '',
  artifact_path  TEXT NOT NULL DEFAULT '',
  trace_path     TEXT NOT NULL DEFAULT '',
  temperature    REAL,
  max_tokens     INTEGER,
  ttft_ms        INTEGER NOT NULL DEFAULT 0,
  total_ms       INTEGER NOT NULL DEFAULT 0,
  tokens_in      INTEGER NOT NULL DEFAULT 0,
  tokens_out     INTEGER NOT NULL DEFAULT 0,
  tokens_think   INTEGER NOT NULL DEFAULT 0,
  tps            REAL NOT NULL DEFAULT 0,
  cost_usd       REAL,                      -- NULL: no price known when it ran
  cost_source    TEXT NOT NULL DEFAULT '',  -- server | price | builtin
  created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS human_scores (
  result_id  INTEGER PRIMARY KEY REFERENCES results(id) ON DELETE CASCADE,
  score      INTEGER NOT NULL CHECK (score BETWEEN 0 AND 10),
  comment    TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS settings (
  key        TEXT PRIMARY KEY,
  value      TEXT NOT NULL DEFAULT '',
  updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

-- USD per million tokens, keyed by the model id as the server reports it.
CREATE TABLE IF NOT EXISTS model_prices (
  model           TEXT PRIMARY KEY,
  input_per_mtok  REAL NOT NULL CHECK (input_per_mtok >= 0),
  output_per_mtok REAL NOT NULL CHECK (output_per_mtok >= 0),
  updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id);
CREATE INDEX IF NOT EXISTS idx_results_prompt ON results(prompt_id);
"""

# Columns added after the first release. Applied with ALTER TABLE when missing,
# so an existing DB keeps working. (table, column, definition)
MIGRATIONS: list[tuple[str, str, str]] = [
    ("results", "trace_path", "TEXT NOT NULL DEFAULT ''"),
    ("runs", "parallel", "INTEGER NOT NULL DEFAULT 1"),
    ("results", "cost_usd", "REAL"),
    ("results", "cost_source", "TEXT NOT NULL DEFAULT ''"),
    ("runs", "effort", "TEXT NOT NULL DEFAULT 'default'"),
]
