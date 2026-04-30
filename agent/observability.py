# pyright: strict
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
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any, Literal, TypeVar

from langfuse import observe as _langfuse_observe  # pyright: ignore[reportUnknownVariableType]

from agent.logging_config import log

ObservationType = Literal[
    "generation",
    "embedding",
    "span",
    "agent",
    "tool",
    "chain",
    "retriever",
    "evaluator",
    "guardrail",
]
F = TypeVar("F", bound=Callable[..., Any])

__all__ = [
    "FLUSH_TIMEOUT_S",
    "enrich_current_generation",
    "flush_langfuse",
    "observe",
    "set_current_span_name",
    "trace_session",
]


def observe(*, name: str | None = None, as_type: ObservationType | None = None) -> Callable[[F], F]:
    """Typed wrapper around `langfuse.observe`.

    Returns a `Callable[[F], F]` decorator that preserves the wrapped
    function's signature. `as_type` is passed through to langfuse at runtime
    but does NOT propagate into the wrapped function's type — reflecting the
    SDK's overload set here would re-introduce the stub gap this wrapper
    exists to contain.

    When Langfuse is disabled, returns a passthrough decorator instead of
    invoking `langfuse.observe`. Without this, the OTel batch span
    processor would still record spans on every call and try to ship them
    to an unreachable backend, spamming the logs.
    """
    if not _LANGFUSE_ENABLED:
        return _passthrough_decorator
    return _langfuse_observe(name=name, as_type=as_type)  # pyright: ignore[reportUnknownVariableType]


FLUSH_TIMEOUT_S = 2.0


def _langfuse_enabled() -> bool:
    """True when both Langfuse keys are present AND `LANGFUSE_DISABLED`
    is not set to a truthy value. The opt-out exists so a developer can
    run without a local Langfuse backend without the OTel exporter
    spamming `localhost:3000` connection-refused errors during every
    call.
    """
    if os.getenv("LANGFUSE_DISABLED", "").lower() in {"1", "true", "yes"}:
        return False
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


# Captured once at import: dotenv has already loaded by the time agent.main
# imports this module, and the env doesn't rotate within a process.
_LANGFUSE_ENABLED = _langfuse_enabled()


def _passthrough_decorator[F: Callable[..., Any]](fn: F) -> F:
    """No-op decorator returned by `observe()` when Langfuse is disabled.
    Module-level so it isn't reallocated on every decoration."""
    return fn


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
    except Exception as exc:
        log.warning("langfuse_flush_failed", error=str(exc))


def enrich_current_generation(*, model: str, usage: dict[str, Any] | None) -> None:
    """Attach model name and token usage to the active Langfuse generation span."""
    if not _LANGFUSE_ENABLED:
        return
    try:
        from langfuse import get_client

        get_client().update_current_generation(model=model, usage_details=usage)
    except Exception as exc:
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
    except Exception as exc:
        log.debug("langfuse_session_id_failed", error=str(exc))
        return contextlib.nullcontext()


def set_current_span_name(name: str) -> None:
    """Rename the active Langfuse span to `name`. No-op when disabled."""
    if not _LANGFUSE_ENABLED:
        return
    try:
        from langfuse import get_client

        get_client().update_current_span(name=name)
    except Exception as exc:
        log.debug("langfuse_span_rename_failed", error=str(exc))
