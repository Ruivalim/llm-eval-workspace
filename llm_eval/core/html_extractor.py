"""Pull a self-contained HTML document out of a model answer that may wrap it
in markdown fences or prose."""

from __future__ import annotations

import re

_FENCE = re.compile(r"```[a-zA-Z0-9_-]*\n([\s\S]*?)```")


def extract_code(answer: str) -> str:
    """First fenced block if any, else the whole answer, stripped."""
    m = _FENCE.search(answer)
    return (m.group(1) if m else answer).strip()


def extract_html(answer: str) -> str:
    """Prefer a fenced block; otherwise slice from <!DOCTYPE or <html to </html>."""
    code = extract_code(answer)
    low = code.lower()
    start = low.find("<!doctype")
    if start < 0:
        start = low.find("<html")
    end = low.rfind("</html>")
    if start >= 0 and end > start:
        return code[start : end + len("</html>")]
    return code
