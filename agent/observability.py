"""Langfuse observability wiring.

Langfuse reads credentials from env vars:
  - LANGFUSE_PUBLIC_KEY
  - LANGFUSE_SECRET_KEY
  - LANGFUSE_HOST (defaults to the cloud host if unset)

If neither key is set, `langfuse_callbacks()` returns an empty list and the
runner falls back to local-only tracing. This keeps Langfuse strictly opt-in.
"""

from __future__ import annotations

import os
from typing import Any

from agent.logging_config import log


def _langfuse_enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def langfuse_callbacks() -> list[Any]:
    """Return a `[CallbackHandler]` when Langfuse env is set, else `[]`."""
    if not _langfuse_enabled():
        log.info("langfuse_disabled", reason="env_missing")
        return []

    # Import lazily: an installed-but-unused langfuse shouldn't force env reads
    # at module import time.
    from langfuse.langchain import CallbackHandler

    handler = CallbackHandler()
    log.info("langfuse_enabled", host=os.getenv("LANGFUSE_HOST", "<default>"))
    return [handler]


def flush_langfuse() -> None:
    """Flush pending Langfuse events. Safe to call when Langfuse is disabled.

    Langfuse batches spans and flushes on a timer; if the process tears down
    between turns we lose the tail. `GraphRunner.stop()` calls this on every
    transport-disconnect so per-call traces always land.
    """
    if not _langfuse_enabled():
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception as exc:  # noqa: BLE001
        log.warning("langfuse_flush_failed", error=str(exc))
