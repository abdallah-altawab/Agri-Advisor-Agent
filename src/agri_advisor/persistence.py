"""Conversation persistence (LangGraph checkpointer backed by SQLite)."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


@contextmanager
def open_checkpointer(db_path: str | Path) -> Iterator[SqliteSaver]:
    """Open a SQLite-backed checkpointer and always close the connection on exit."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    try:
        yield SqliteSaver(conn)
    finally:
        conn.close()