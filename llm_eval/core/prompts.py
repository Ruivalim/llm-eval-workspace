"""Prompt library: YAML files in data/prompts, one list per category, each
prompt carrying every locale. `en` is mandatory and the fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from llm_eval.config import DEFAULT_LOCALE, PROMPTS_DIR

EvalType = Literal["text", "render_html", "tool_call", "error_test"]


class LocaleText(BaseModel):
    name: str
    system_prompt: str = ""
    prompt: str


class Check(BaseModel):
    """Spec of the automatic check. `type` selects the checker in core/checks.py;
    the remaining keys are its parameters."""

    type: str = "none"
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _split(cls, data: Any) -> Any:
        if isinstance(data, dict) and "params" not in data:
            data = dict(data)
            return {"type": data.pop("type", "none"), "params": data}
        return data


class Prompt(BaseModel):
    id: str
    category: str
    eval_type: EvalType
    max_tokens: int | None = None  # None: do not send it, the server default applies
    tags: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    fault: dict[str, Any] | None = None
    check: Check = Field(default_factory=Check)
    locales: dict[str, LocaleText]

    @model_validator(mode="after")
    def _needs_en(self) -> Prompt:
        if DEFAULT_LOCALE not in self.locales:
            raise ValueError(f"prompt {self.id!r} lacks the {DEFAULT_LOCALE!r} locale")
        return self

    def text(self, locale: str = DEFAULT_LOCALE) -> LocaleText:
        return self.locales.get(locale) or self.locales[DEFAULT_LOCALE]

    @property
    def available_locales(self) -> list[str]:
        return sorted(self.locales)


class Library(BaseModel):
    prompts: list[Prompt]

    def by_id(self, prompt_id: str) -> Prompt:
        for p in self.prompts:
            if p.id == prompt_id:
                return p
        raise KeyError(prompt_id)

    @property
    def categories(self) -> list[str]:
        seen: dict[str, None] = {}
        for p in self.prompts:
            seen.setdefault(p.category, None)
        return list(seen)

    def in_category(self, category: str) -> list[Prompt]:
        return [p for p in self.prompts if p.category == category]


def load_library(directory: Path = PROMPTS_DIR) -> Library:
    prompts: list[Prompt] = []
    ids: set[str] = set()
    for file in sorted(directory.glob("*.yaml")):
        raw = yaml.safe_load(file.read_text(encoding="utf-8")) or []
        if not isinstance(raw, list):
            raise ValueError(f"{file.name}: expected a top-level list")
        for item in raw:
            prompt = Prompt.model_validate(item)
            if prompt.id in ids:
                raise ValueError(f"duplicate prompt id {prompt.id!r} in {file.name}")
            ids.add(prompt.id)
            prompts.append(prompt)
    return Library(prompts=prompts)
