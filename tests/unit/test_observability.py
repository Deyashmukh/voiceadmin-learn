"""Unit tests for `agent.observability` — Langfuse helper wrappers.

The helpers do function-local imports (`from langfuse import ...` inside each
function body) so the disabled path can short-circuit without paying the
import cost; the tests rely on that contract — patching `sys.modules` only
works because the import happens at call time, not module load. Don't hoist
the imports without updating these tests.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from unittest.mock import MagicMock, patch

import pytest

from agent import observability


@pytest.mark.parametrize("enabled", [True, False])
async def test_flush_langfuse_calls_sdk_only_when_enabled(monkeypatch, enabled):
    fake_client = MagicMock()
    fake_get_client = MagicMock(return_value=fake_client)
    monkeypatch.setattr(observability, "_LANGFUSE_ENABLED", enabled)
    with patch.dict("sys.modules", {"langfuse": MagicMock(get_client=fake_get_client)}):
        await observability.flush_langfuse()

    if enabled:
        fake_client.flush.assert_called_once()
    else:
        fake_get_client.assert_not_called()


@pytest.mark.parametrize("enabled", [True, False])
def test_enrich_current_generation_calls_sdk_only_when_enabled(monkeypatch, enabled):
    fake_client = MagicMock()
    fake_get_client = MagicMock(return_value=fake_client)
    monkeypatch.setattr(observability, "_LANGFUSE_ENABLED", enabled)
    with patch.dict("sys.modules", {"langfuse": MagicMock(get_client=fake_get_client)}):
        observability.enrich_current_generation(
            model="claude-haiku-4-5", usage={"input": 10, "output": 5}
        )

    if enabled:
        fake_client.update_current_generation.assert_called_once_with(
            model="claude-haiku-4-5", usage_details={"input": 10, "output": 5}
        )
    else:
        fake_get_client.assert_not_called()


@pytest.mark.parametrize("enabled", [True, False])
def test_set_current_span_name_calls_sdk_only_when_enabled(monkeypatch, enabled):
    fake_client = MagicMock()
    fake_get_client = MagicMock(return_value=fake_client)
    monkeypatch.setattr(observability, "_LANGFUSE_ENABLED", enabled)
    with patch.dict("sys.modules", {"langfuse": MagicMock(get_client=fake_get_client)}):
        observability.set_current_span_name("tool_dispatch.send_dtmf")

    if enabled:
        fake_client.update_current_span.assert_called_once_with(name="tool_dispatch.send_dtmf")
    else:
        fake_get_client.assert_not_called()


@pytest.mark.parametrize("enabled", [True, False])
def test_trace_session_propagates_session_id_only_when_enabled(monkeypatch, enabled):
    fake_cm = MagicMock(spec=AbstractContextManager)
    fake_propagate = MagicMock(return_value=fake_cm)
    monkeypatch.setattr(observability, "_LANGFUSE_ENABLED", enabled)
    with patch.dict("sys.modules", {"langfuse": MagicMock(propagate_attributes=fake_propagate)}):
        cm = observability.trace_session("CA-abc")

    assert isinstance(cm, AbstractContextManager)
    if enabled:
        fake_propagate.assert_called_once_with(session_id="CA-abc")
        assert cm is fake_cm
    else:
        fake_propagate.assert_not_called()
        # Disabled path returns nullcontext; using it must not raise.
        with cm:
            pass


def test_trace_session_falls_back_to_nullcontext_on_sdk_error(monkeypatch):
    """If the langfuse call raises, the wrapper degrades to `nullcontext()`
    so callers' `with` blocks still work — observability failures must not
    take down a call."""
    boom = MagicMock(side_effect=RuntimeError("simulated"))
    monkeypatch.setattr(observability, "_LANGFUSE_ENABLED", True)
    with patch.dict("sys.modules", {"langfuse": MagicMock(propagate_attributes=boom)}):
        cm = observability.trace_session("CA-x")

    assert isinstance(cm, AbstractContextManager)
    with cm:
        pass


def test_set_current_span_name_swallows_sdk_errors(monkeypatch):
    fake_client = MagicMock()
    fake_client.update_current_span.side_effect = RuntimeError("simulated")
    monkeypatch.setattr(observability, "_LANGFUSE_ENABLED", True)
    with patch.dict(
        "sys.modules", {"langfuse": MagicMock(get_client=MagicMock(return_value=fake_client))}
    ):
        observability.set_current_span_name("tool_dispatch.send_dtmf")  # must not raise


def test_enrich_current_generation_swallows_sdk_errors(monkeypatch):
    fake_client = MagicMock()
    fake_client.update_current_generation.side_effect = RuntimeError("simulated")
    monkeypatch.setattr(observability, "_LANGFUSE_ENABLED", True)
    with patch.dict(
        "sys.modules", {"langfuse": MagicMock(get_client=MagicMock(return_value=fake_client))}
    ):
        observability.enrich_current_generation(model="x", usage=None)  # must not raise
