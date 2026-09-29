import json

from llm_eval.core.tool_mock import execute, schemas_for


def test_calculate_math_is_exact_and_safe():
    assert json.loads(execute("calculate_math", {"expression": "1284 * 43"}))["result"] == "55212"
    assert json.loads(execute("calculate_math", {"expression": "1284 x 43"}))["result"] == "55212"
    assert "error" in json.loads(execute("calculate_math", {"expression": "__import__('os')"}))


def test_control_light_succeeds_or_fails_cleanly():
    assert json.loads(execute("control_light", {"room": "kitchen", "action": "on"})) == {
        "room": "kitchen",
        "state": "on",
    }
    assert (
        "offline"
        in json.loads(execute("control_light", {"room": "garage", "action": "on"}))["error"]
    )
    assert "error" in json.loads(execute("control_light", {"room": "attic", "action": "on"}))


def test_unknown_tool_reports_error():
    assert "error" in json.loads(execute("nope", {}))


def test_schemas_for_ignores_unknown():
    assert [s["function"]["name"] for s in schemas_for(["get_weather", "zzz"])] == ["get_weather"]
