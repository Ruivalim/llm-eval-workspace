from llm_eval.core.html_extractor import extract_code, extract_html


def test_extract_code_prefers_first_fence():
    assert extract_code("text\n```html\n<b>a</b>\n```\nmore\n```\nz\n```") == "<b>a</b>"
    assert extract_code("no fence") == "no fence"


def test_extract_html_slices_document_out_of_prose():
    ans = "Here you go:\n<!DOCTYPE html><html><body>hi</body></html>\nHope it helps!"
    assert extract_html(ans) == "<!DOCTYPE html><html><body>hi</body></html>"


def test_extract_html_returns_fragment_untouched():
    assert extract_html("<div>x</div>") == "<div>x</div>"
