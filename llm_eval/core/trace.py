"""Execution trace: one JSON Lines file per evaluation with everything needed
to replay it later. Request (full messages, params, tools), every raw chunk the
server sent, tool calls and results, telemetry, verdict, errors.

Files land in `data/traces/YYYY-MM-DD/<HHMMSS.mmm>__<prompt>__<model>.jsonl`.
Each line: {"t_ms": <ms since trace start>, "ts": <ISO UTC>, "ev": <kind>, ...}."""

from __future__ import annotations

import json
import re
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")[:80] or "x"


class Trace:
    """Append-only writer. Safe to call from one producer thread; the API's
    evaluate thread is the only writer per file."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._t0 = time.perf_counter()
        self._lock = threading.Lock()
        self._fh = self.path.open("a", encoding="utf-8")

    @classmethod
    def start(cls, directory: Path, prompt_id: str, model: str) -> Trace:
        now = datetime.now(UTC)
        name = (
            f"{now:%H%M%S}.{now.microsecond // 1000:03d}__{_slug(prompt_id)}__{_slug(model)}.jsonl"
        )
        return cls(directory / f"{now:%Y-%m-%d}" / name)

    def event(self, ev: str, **data: Any) -> None:
        line = {
            "t_ms": round((time.perf_counter() - self._t0) * 1000, 2),
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "ev": ev,
            **data,
        }
        with self._lock:
            self._fh.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()

    def __enter__(self) -> Trace:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class NullTrace:
    """Same surface, writes nothing. Lets callers skip `if trace:` checks."""

    path: Path | None = None

    def event(self, ev: str, **data: Any) -> None:
        return None

    def close(self) -> None:
        return None

    def __enter__(self) -> NullTrace:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


TraceLike = Trace | NullTrace


def read_trace(path: Path) -> list[dict[str, Any]]:
    """Parse a trace file back into dicts. Skips corrupt lines instead of failing."""
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            out.append({"ev": "corrupt_line", "raw": line[:200]})
    return out
