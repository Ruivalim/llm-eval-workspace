"""Unsloth Studio extras on top of the OpenAI-compatible surface: `/models`
returns a `loaded` flag and `POST /api/inference/load` swaps the active model.
Only used when the server flavor is `unsloth`."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import httpx


def _origin(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    return base_url[:-3] if base_url.endswith("/v1") else base_url


def _headers(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def models(
    base_url: str,
    api_key: str | None,
    timeout_s: float = 30,
    client: httpx.Client | None = None,
) -> list[dict[str, Any]]:
    http = client or httpx
    r = http.get(f"{base_url.rstrip('/')}/models", headers=_headers(api_key), timeout=timeout_s)
    r.raise_for_status()
    return r.json().get("data", [])


def active(
    base_url: str, api_key: str | None, client: httpx.Client | None = None
) -> dict[str, Any] | None:
    return next((m for m in models(base_url, api_key, client=client) if m.get("loaded")), None)


def load(
    base_url: str,
    api_key: str | None,
    model_id: str,
    wait_s: float = 600,
    poll_s: float = 5,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Ask the server to load `model_id`, then poll until it reports loaded.
    Raises TimeoutError after wait_s; HTTP errors propagate as httpx errors."""
    http = client or httpx
    r = http.post(
        f"{_origin(base_url)}/api/inference/load",
        headers=_headers(api_key),
        json={"model_path": model_id},
        timeout=300,
    )
    r.raise_for_status()
    deadline = clock() + wait_s
    while True:
        cur = active(base_url, api_key, client=client)
        if cur and cur.get("id") == model_id:
            return
        if clock() >= deadline:
            raise TimeoutError(f"{model_id} not active after {wait_s}s")
        sleep(poll_s)
