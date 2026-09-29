import pytest

from llm_eval.db.connection import connect


@pytest.fixture
def conn():
    c = connect(":memory:")
    yield c
    c.close()


@pytest.fixture(autouse=True)
def no_backoff_sleep(monkeypatch):
    """Retries back off for real seconds; tests record the waits instead."""
    waits: list[float] = []

    def fake(seconds, hook):
        waits.append(seconds)
        return hook.cancelled if hook is not None else False

    monkeypatch.setattr("llm_eval.core.client._sleep", fake)
    return waits
