"""Application log: stderr for the terminal, plus a rotating file in
`data/logs/llm_eval.log` so a session can be reconstructed afterwards.
Per-evaluation detail (every chunk) goes to `core/trace.py`, not here."""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-5s %(name)s: %(message)s"
_configured = False


def configure(logs_dir: Path, level: str | None = None) -> None:
    global _configured
    if _configured:
        return
    _configured = True
    lvl = (level or os.getenv("LLM_EVAL_LOG_LEVEL") or "INFO").upper()
    root = logging.getLogger("llm_eval")
    root.setLevel(lvl)
    fmt = logging.Formatter(_FORMAT)
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    root.addHandler(stream)
    logs_dir.mkdir(parents=True, exist_ok=True)
    rotating = logging.handlers.RotatingFileHandler(
        logs_dir / "llm_eval.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    rotating.setFormatter(fmt)
    root.addHandler(rotating)
    root.propagate = False
    root.debug("logging to %s (level %s)", logs_dir, lvl)
