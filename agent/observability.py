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
    """Drain Langfuse's batch buffer so the tail of a call doesn't get lost on shutdown."""
    if not _langfuse_enabled():
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception as exc:  # noqa: BLE001
        log.warning("langfuse_flush_failed", error=str(exc))


def enrich_current_generation(*, model: str, usage: dict[str, Any] | None) -> None:
    """Attach model name and token usage to the active Langfuse generation span.

    No-op when Langfuse is disabled — the stub client swallows the call.
    """
    try:
        from langfuse import get_client

        get_client().update_current_generation(model=model, usage_details=usage)
    except Exception as exc:  # noqa: BLE001
        log.debug("langfuse_enrich_failed", error=str(exc))
