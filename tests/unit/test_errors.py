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


def test_actuator_error_carries_intent_kind():
    err = ActuatorError("twilio_client missing", intent_kind="dtmf")
    assert err.intent_kind == "dtmf"
    assert "twilio_client missing" in str(err)


def test_configuration_error_is_plain_message_only():
    """`ConfigurationError` is intentionally not parameterized — it's a
    construction-time failure, not a runtime correlation."""
    err = ConfigurationError("ANTHROPIC_API_KEY is not set")
    assert isinstance(err, AgentError)
    assert str(err) == "ANTHROPIC_API_KEY is not set"


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
        lambda: ConfigurationError("x"),
        lambda: LLMRefusalError("x", stop_reason="r", response_id="i"),
        lambda: ToolDispatchError("x", tool_name="speak"),
        lambda: ActuatorError("x", intent_kind="speak"),
        lambda: DestinationNotAllowedError("x"),
    ],
)
def test_every_error_can_be_raised_and_caught(factory: Callable[[], AgentError]):
    """Smoke-test the catch flow that the E2 migration will rely on."""
    with pytest.raises(AgentError):
        raise factory()
