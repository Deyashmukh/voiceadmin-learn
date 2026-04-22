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


def langfuse_callbacks() -> list[Any]:
    """Return a `[CallbackHandler]` when Langfuse env is set, else `[]`."""
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        log.info("langfuse_disabled", reason="env_missing")
        return []

    # Import lazily: an installed-but-unused langfuse shouldn't force env reads
    # at module import time.
    from langfuse.langchain import CallbackHandler

    handler = CallbackHandler()
    log.info("langfuse_enabled", host=os.getenv("LANGFUSE_HOST", "<default>"))
    return [handler]
