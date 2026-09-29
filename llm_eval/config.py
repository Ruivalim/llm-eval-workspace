"""Environment and paths. Everything configurable lives here, read once."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROMPTS_DIR = DATA_DIR / "prompts"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
TRACES_DIR = DATA_DIR / "traces"
LOGS_DIR = DATA_DIR / "logs"

DEFAULT_LOCALE = "en"
SUPPORTED_LOCALES = ("en", "pt-BR")


def _flag(name: str) -> bool:
    """Env boolean: unset, empty and "0" are off; anything else is on."""
    return os.getenv(name, "").strip().lower() not in ("", "0", "false", "no")


@dataclass(frozen=True)
class Settings:
    db_path: Path
    default_base_url: str
    default_api_key: str | None
    locale: str
    artifacts_dir: Path = ARTIFACTS_DIR
    traces_dir: Path = TRACES_DIR
    logs_dir: Path = LOGS_DIR
    default_flavor: str = "openai"
    allow_code_exec: bool = False


def load_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    return Settings(
        db_path=Path(os.getenv("LLM_EVAL_DB", DATA_DIR / "llm_eval.db")),
        # No default endpoint on purpose: the first run should show the UI's own
        # starter suggestions, not somebody else's server.
        default_base_url=os.getenv("LLM_EVAL_DEFAULT_BASE_URL", "").strip(),
        default_api_key=os.getenv("LLM_EVAL_DEFAULT_API_KEY") or None,
        locale=os.getenv("LLM_EVAL_LOCALE", DEFAULT_LOCALE),
        artifacts_dir=Path(os.getenv("LLM_EVAL_ARTIFACTS", ARTIFACTS_DIR)),
        traces_dir=Path(os.getenv("LLM_EVAL_TRACES", TRACES_DIR)),
        logs_dir=Path(os.getenv("LLM_EVAL_LOGS", LOGS_DIR)),
        default_flavor=os.getenv("LLM_EVAL_DEFAULT_FLAVOR", "openai"),
        allow_code_exec=_flag("LLM_EVAL_ALLOW_CODE_EXEC"),
    )
