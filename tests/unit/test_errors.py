# pyright: strict
"""Unit tests for `agent.errors`. Locks the typed-error contract so
catch-by-type at call sites stays correct as message text drifts."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from agent.errors import (
    ActuatorError,
    AgentError,
    ConfigurationError,
    DestinationNotAllowedError,
    LLMRefusalError,
    ToolDispatchError,
)
from agent.schemas import IntentKind


def test_subclasses_descend_from_agent_error():
    """Catching `AgentError` must catch every typed agent error so callers
    can distinguish agent-internal failures from infrastructure errors."""
    for cls in (
        ConfigurationError,
        LLMRefusalError,
        ToolDispatchError,
        ActuatorError,
        DestinationNotAllowedError,
    ):
        assert issubclass(cls, AgentError)


def test_agent_error_is_a_proper_exception():
    err = AgentError("something broke")
    assert isinstance(err, Exception)
    assert str(err) == "something broke"


def test_llm_refusal_error_carries_correlation_fields():
    err = LLMRefusalError("no parsed_output", stop_reason="refusal", response_id="msg_abc123")
    assert isinstance(err, AgentError)
    assert err.stop_reason == "refusal"
    assert err.response_id == "msg_abc123"
    assert "no parsed_output" in str(err)


def test_tool_dispatch_error_carries_tool_name():
    err = ToolDispatchError("handler crashed", tool_name="send_dtmf")
    assert err.tool_name == "send_dtmf"
    assert "handler crashed" in str(err)


def test_tool_dispatch_error_accepts_unknown_tool():
    """Malformed-frame case: dispatcher couldn't resolve the tool name."""
    err = ToolDispatchError("malformed call frame", tool_name=None)
    assert err.tool_name is None


@pytest.mark.parametrize("kind", ["dtmf", "speak", "hangup"])
def test_actuator_error_accepts_every_intent_kind(kind: IntentKind):
    """Every value in the `IntentKind` Literal must construct cleanly —
    if `agent.schemas.IntentKind` adds a variant without an
    `actuator.execute` handler, this test forces the test author to
    decide whether `ActuatorError` should also raise for it."""
    err = ActuatorError("boom", intent_kind=kind)
    assert err.intent_kind == kind


def test_configuration_error_carries_setting_name():
    err = ConfigurationError("ANTHROPIC_API_KEY is not set", setting="ANTHROPIC_API_KEY")
    assert isinstance(err, AgentError)
    assert err.setting == "ANTHROPIC_API_KEY"
    assert "ANTHROPIC_API_KEY is not set" in str(err)


def test_destination_not_allowed_carries_destination():
    err = DestinationNotAllowedError(
        "Destination '+15551112222' is not in ALLOWED_DESTINATIONS",
        destination="+15551112222",
    )
    assert err.destination == "+15551112222"
    assert "+15551112222" in str(err)


def test_destination_not_allowed_re_export_is_same_class():
    """Existing imports from `agent.telephony.dialer` must reference the
    same class as the canonical one in `agent.errors` — otherwise
    `except DestinationNotAllowedError:` blocks would miss raises that
    use the other name."""
    from agent.telephony.dialer import (
        DestinationNotAllowedError as DialerReexport,
    )

    assert DialerReexport is DestinationNotAllowedError


@pytest.mark.parametrize(
    "factory",
    [
        lambda: AgentError("x"),
        lambda: ConfigurationError("x", setting="X"),
        lambda: LLMRefusalError("x", stop_reason="refusal", response_id="i"),
        lambda: ToolDispatchError("x", tool_name="speak"),
        lambda: ActuatorError("x", intent_kind="speak"),
        lambda: DestinationNotAllowedError("x", destination="+15555555555"),
    ],
)
def test_every_error_can_be_raised_and_caught_as_agent_error(
    factory: Callable[[], AgentError],
):
    """Every typed error can be raised and caught as `AgentError`."""
    with pytest.raises(AgentError):
        raise factory()


def test_subclass_can_be_caught_specifically():
    """The whole point of the taxonomy: `except SpecificError:` only catches
    its variant, not other AgentError subclasses. Locks the discriminator."""
    with pytest.raises(LLMRefusalError):
        raise LLMRefusalError("x", stop_reason="refusal", response_id="i")

    # An LLMRefusalError must NOT be caught by a different subclass's handler.
    with pytest.raises(AgentError):  # outer guard so the test fails loudly
        try:
            raise LLMRefusalError("x", stop_reason="refusal", response_id="i")
        except ActuatorError:  # pragma: no cover (would mean MRO is broken)
            pytest.fail("LLMRefusalError caught by ActuatorError handler")
