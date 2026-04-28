"""Langfuse observability wiring.

Langfuse reads credentials from env vars:
  - LANGFUSE_PUBLIC_KEY
  - LANGFUSE_SECRET_KEY
  - LANGFUSE_HOST (defaults to the cloud host if unset)

When Langfuse is unavailable (env missing or backend unreachable), every
helper here is a quiet no-op — the agent runs without traces but doesn't
fail. `@observe` decorators elsewhere are likewise no-ops without keys
(the SDK's `get_client()` returns a stub).

Known UI quirk: Langfuse's `@observe` records `asyncio.CancelledError` as
an ERROR-level span. Barge-ins show up as red error traces in the UI even
though they're a normal re-do, not a failure. The SDK overwrites any
manual `level=DEFAULT` set inside the wrapped function, so the fix is
either an upstream change to Langfuse or replacing `@observe` with a
custom wrapper — deferred.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from contextlib import AbstractContextManager
from typing import Any

from agent.logging_config import log

FLUSH_TIMEOUT_S = 2.0
# Captured once at import: dotenv has already loaded by the time agent.main
# imports this module, and the keys don't rotate within a process.
_LANGFUSE_ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


async def flush_langfuse() -> None:
    """Drain Langfuse's batch buffer so the tail of a call doesn't get lost on shutdown.

    `Langfuse.flush()` is sync and chains into OTel's `force_flush(timeout=30s)`,
    so on transport-disconnect with an unreachable Langfuse it could stall the
    teardown path. Run it on a worker thread with a 2s budget — plenty for a
    healthy backend, won't block when it isn't.
    """
    if not _LANGFUSE_ENABLED:
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
    if not _LANGFUSE_ENABLED:
        return
    try:
        from langfuse import get_client

        get_client().update_current_generation(model=model, usage_details=usage)
    except Exception as exc:  # noqa: BLE001
        log.debug("langfuse_enrich_failed", error=str(exc))


def trace_session(call_sid: str) -> AbstractContextManager[Any]:
    """Tag every Langfuse span within this context with `session_id=call_sid`.
    Groups all turns of one call under a single session in the Langfuse UI
    rather than N isolated traces. No-op when Langfuse is disabled.
    """
    if not _LANGFUSE_ENABLED:
        return contextlib.nullcontext()
    try:
        from langfuse import propagate_attributes

        return propagate_attributes(session_id=call_sid)
    except Exception as exc:  # noqa: BLE001
        log.debug("langfuse_session_id_failed", error=str(exc))
        return contextlib.nullcontext()


def set_current_span_name(name: str) -> None:
    """Rename the active Langfuse span to `name`. No-op when disabled."""
    if not _LANGFUSE_ENABLED:
        return
    try:
        from langfuse import get_client

        get_client().update_current_span(name=name)
    except Exception as exc:  # noqa: BLE001
        log.debug("langfuse_span_rename_failed", error=str(exc))
