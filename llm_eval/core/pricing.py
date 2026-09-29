"""What one evaluation cost, in USD.

Three sources, first match wins:
1. `server`: the endpoint said so in usage.cost (OpenRouter and similar).
2. `price`: tokens times the per-model price set in Settings.
3. `builtin`: tokens times Anthropic's list price below, for Claude ids.

No match means unknown (None), which is the honest answer for a local model
nobody priced. Input is billed at the plain input rate: bench prompts are far
below the minimum cacheable prefix, so cache discounts would not apply anyway.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

from llm_eval.core.telemetry import Telemetry

# USD per million tokens, Anthropic first-party API. Bedrock and Vertex bill
# differently; set a price in Settings to override any of these.
BUILTIN_AS_OF = "2026-09-25"
BUILTIN: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

_DATED = re.compile(r"-\d{8}$")


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float
    source: str  # "price" | "builtin"

    def cost(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * self.input_per_mtok + tokens_out * self.output_per_mtok) / 1_000_000


def builtin_price(model: str) -> Price | None:
    """Exact id, or the id with a dated snapshot suffix (claude-haiku-4-5-20251001).
    Never a looser prefix: claude-opus-5 must not price claude-opus-5-5."""
    hit = BUILTIN.get(model) or BUILTIN.get(_DATED.sub("", model))
    return Price(hit[0], hit[1], "builtin") if hit else None


def user_price(conn: sqlite3.Connection, model: str) -> Price | None:
    row = conn.execute(
        "SELECT input_per_mtok, output_per_mtok FROM model_prices WHERE model = ?", (model,)
    ).fetchone()
    return Price(row[0], row[1], "price") if row else None


def resolve(conn: sqlite3.Connection, model: str) -> Price | None:
    return user_price(conn, model) or builtin_price(model)


def cost_of(conn: sqlite3.Connection, model: str, telemetry: Telemetry) -> tuple[float | None, str]:
    """(cost in USD or None when unknown, source)."""
    if telemetry.cost_reported is not None:
        return float(telemetry.cost_reported), "server"
    price = resolve(conn, model)
    if price is None:
        return None, ""
    return price.cost(telemetry.tokens_in, telemetry.tokens_out), price.source
