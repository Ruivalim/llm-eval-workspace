"""Create the DB and seed a default server from the environment, when one is set.
Prompts live in data/prompts and are not copied into the DB.
`python -m llm_eval.db.init_db --reset`."""

from __future__ import annotations

import argparse
import sqlite3

from llm_eval.config import Settings, load_settings
from llm_eval.db import crud
from llm_eval.db.connection import connect


def seed_default_server(conn: sqlite3.Connection, settings: Settings) -> None:
    """Only seed when the environment asks for it: a fresh install should not show
    an unreachable server somebody else configured."""
    if crud.list_servers(conn) or not settings.default_base_url:
        return
    crud.add_server(
        conn,
        name="default",
        base_url=settings.default_base_url,
        api_key=settings.default_api_key,
        flavor=settings.default_flavor,
    )


def init(settings: Settings, reset: bool = False) -> sqlite3.Connection:
    if reset and settings.db_path.exists():
        settings.db_path.unlink()
    conn = connect(settings.db_path)
    seed_default_server(conn, settings)
    return conn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="delete the DB first")
    args = parser.parse_args()
    settings = load_settings()
    init(settings, reset=args.reset)
    print(f"db ready at {settings.db_path}")


if __name__ == "__main__":
    main()
