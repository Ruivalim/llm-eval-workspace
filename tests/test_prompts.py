from pathlib import Path

import pytest
import yaml

from llm_eval.config import PROMPTS_DIR, SUPPORTED_LOCALES
from llm_eval.core.checks import CHECKERS
from llm_eval.core.prompts import Prompt, load_library


def test_library_loads_every_prompt_with_english():
    lib = load_library(PROMPTS_DIR)
    assert len(lib.prompts) >= 14
    for p in lib.prompts:
        assert "en" in p.locales, p.id
        assert p.text("en").prompt.strip()


def test_every_check_type_is_registered():
    lib = load_library(PROMPTS_DIR)
    for p in lib.prompts:
        assert p.check.type in CHECKERS, f"{p.id}: unknown check {p.check.type}"


def test_tool_prompts_declare_tools_that_the_check_expects():
    lib = load_library(PROMPTS_DIR)
    for p in lib.prompts:
        if p.eval_type == "tool_call":
            assert p.tools, p.id
            for t in p.check.params.get("tools", []):
                assert t in p.tools, f"{p.id}: check expects {t} but prompt does not offer it"


def test_categories_file_covers_every_category():
    cats = {c["slug"] for c in yaml.safe_load((PROMPTS_DIR.parent / "categories.yaml").read_text())}
    lib = load_library(PROMPTS_DIR)
    assert set(lib.categories) <= cats


def test_only_supported_locales_are_used():
    lib = load_library(PROMPTS_DIR)
    for p in lib.prompts:
        assert set(p.locales) <= set(SUPPORTED_LOCALES), p.id


def test_locale_fallback_to_english():
    p = Prompt.model_validate(
        {
            "id": "x",
            "category": "c",
            "eval_type": "text",
            "locales": {"en": {"name": "n", "prompt": "hi"}},
        }
    )
    assert p.text("pt-BR").prompt == "hi"
    assert p.check.type == "none"


def test_missing_english_is_rejected():
    with pytest.raises(ValueError):
        Prompt.model_validate(
            {
                "id": "x",
                "category": "c",
                "eval_type": "text",
                "locales": {"pt-BR": {"name": "n", "prompt": "oi"}},
            }
        )


def test_duplicate_ids_are_rejected(tmp_path: Path):
    body = "- id: dup\n  category: c\n  eval_type: text\n  locales: {en: {name: a, prompt: b}}\n"
    (tmp_path / "a.yaml").write_text(body)
    (tmp_path / "b.yaml").write_text(body)
    with pytest.raises(ValueError, match="duplicate"):
        load_library(tmp_path)


def test_check_params_split_from_type():
    p = Prompt.model_validate(
        {
            "id": "x",
            "category": "c",
            "eval_type": "text",
            "check": {"type": "final_number", "expected": 4},
            "locales": {"en": {"name": "n", "prompt": "hi"}},
        }
    )
    assert p.check.type == "final_number"
    assert p.check.params == {"expected": 4}
