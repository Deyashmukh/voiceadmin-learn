"""Langfuse observability wiring.

Langfuse reads credentials from env vars:
  - LANGFUSE_PUBLIC_KEY
  - LANGFUSE_SECRET_KEY
  - LANGFUSE_HOST (defaults to the cloud host if unset)

If neither key is set, `langfuse_callbacks()` returns an empty list and the
runner falls back to local-only tracing. This keeps Langfuse strictly opt-in.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from agent.logging_config import log

FLUSH_TIMEOUT_S = 2.0


def _langfuse_enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def langfuse_callbacks() -> list[BaseCallbackHandler]:
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


async def flush_langfuse() -> None:
    """Drain Langfuse's batch buffer so the tail of a call doesn't get lost on shutdown.

    `Langfuse.flush()` is sync and chains into OTel's `force_flush(timeout=30s)`,
    so on transport-disconnect with an unreachable Langfuse it could stall the
    teardown path. Run it on a worker thread with a 2s budget — plenty for a
    healthy backend, won't block when it isn't.
    """
    if not _langfuse_enabled():
        return
    try:
        from langfuse import get_client

        await asyncio.wait_for(asyncio.to_thread(get_client().flush), timeout=FLUSH_TIMEOUT_S)
    except TimeoutError:
        log.warning("langfuse_flush_timeout", timeout_s=FLUSH_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001
        log.warning("langfuse_flush_failed", error=str(exc))


def enrich_current_generation(*, model: str, usage: dict[str, Any] | None) -> None:
    """Attach model name and token usage to the active Langfuse generation span."""
    if not _langfuse_enabled():
        return
    try:
        from langfuse import get_client

        get_client().update_current_generation(model=model, usage_details=usage)
    except Exception as exc:  # noqa: BLE001
        log.debug("langfuse_enrich_failed", error=str(exc))
