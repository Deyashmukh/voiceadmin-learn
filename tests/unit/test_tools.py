# pyright: strict
"""Unit tests for the IVR tools dispatcher + arg validation."""

from __future__ import annotations

import time
from typing import ClassVar
from unittest.mock import MagicMock

import pytest
import structlog.testing

from agent import tools
from agent.schemas import (
    DTMFIntent,
    HangupIntent,
    SpeakIntent,
    ToolCall,
)
from agent.telephony.dtmf import send_digits

from .conftest import MakeSession

# --- Structural arg validation (delegates to Pydantic) ----------------------


async def test_dispatch_send_dtmf_with_invalid_args_rejected(make_session: MakeSession):
    """Schema-level rejection: pattern mismatch."""
    s = make_session()
    result = await tools.dispatch(ToolCall(name="send_dtmf", args={"digits": "abc"}), s)
    assert not result.success
    assert not result.advanced_call_state


async def test_dispatch_missing_required_arg_rejected(make_session: MakeSession):
    s = make_session()
    result = await tools.dispatch(ToolCall(name="send_dtmf", args={}), s)
    assert not result.success


async def test_dispatch_unknown_tool_name_raises_keyerror(make_session: MakeSession):
    """Locks the contract: if the `ToolName` Literal invariant is bypassed
    (e.g., a dict-built ToolCall from a Langfuse replay or a future SDK that
    doesn't validate names), the dispatcher fails loudly rather than silently
    no-oping. The IVR LLM client is responsible for catching hallucinations
    at the client boundary before they reach here."""

    class _ForgedToolCall:
        name = "made_up"
        args: ClassVar[dict[str, object]] = {}

    s = make_session()
    with pytest.raises(KeyError):
        await tools.dispatch(_ForgedToolCall(), s)  # pyright: ignore[reportArgumentType]


# --- send_dtmf: contextual validation ---------------------------------------


async def test_send_dtmf_with_no_recent_menu_options_accepts_any(make_session: MakeSession):
    """No menu has been seen yet → any digit goes through."""
    s = make_session()
    assert s.recent_menu_options == []
    result = await tools.dispatch(ToolCall(name="send_dtmf", args={"digits": "1"}), s)
    assert result.success
    assert result.advanced_call_state
    assert isinstance(result.side_effect, DTMFIntent)
    assert result.side_effect.digits == "1"


async def test_send_dtmf_with_offered_digit_accepted(make_session: MakeSession):
    s = make_session(recent_menu_options=["1", "2", "3"])
    result = await tools.dispatch(ToolCall(name="send_dtmf", args={"digits": "2"}), s)
    assert result.success
    assert isinstance(result.side_effect, DTMFIntent)


async def test_send_dtmf_with_unoffered_digit_rejected(make_session: MakeSession):
    s = make_session(recent_menu_options=["1", "2", "3"])
    result = await tools.dispatch(ToolCall(name="send_dtmf", args={"digits": "9"}), s)
    assert not result.success
    assert not result.advanced_call_state
    assert "not offered by the most recent menu" in result.message
    assert result.side_effect is None


async def test_send_dtmf_universal_keys_always_allowed(make_session: MakeSession):
    """`#` and `*` are universal — accept regardless of recent menu options."""
    s = make_session(recent_menu_options=["1", "2"])
    result = await tools.dispatch(ToolCall(name="send_dtmf", args={"digits": "#"}), s)
    assert result.success


async def test_send_dtmf_member_id_with_pound_passes_when_offered(make_session: MakeSession):
    """Composite digit string: `123456#`. Each char must be offered or universal."""
    s = make_session(recent_menu_options=["1", "2", "3", "4", "5", "6"])
    result = await tools.dispatch(ToolCall(name="send_dtmf", args={"digits": "123456#"}), s)
    assert result.success
    assert isinstance(result.side_effect, DTMFIntent)


async def test_send_dtmf_composite_with_unoffered_digit_rejected(make_session: MakeSession):
    s = make_session(recent_menu_options=["1", "2", "3"])
    result = await tools.dispatch(ToolCall(name="send_dtmf", args={"digits": "129"}), s)
    assert not result.success
    assert "9" in result.message


# --- speak ------------------------------------------------------------------


async def test_speak_emits_speak_intent(make_session: MakeSession):
    s = make_session()
    result = await tools.dispatch(ToolCall(name="speak", args={"text": "yes, continuing"}), s)
    assert result.success
    assert isinstance(result.side_effect, SpeakIntent)
    assert result.side_effect.text == "yes, continuing"


# --- record_benefit: type-vs-field-type validation --------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("active", True),
        ("active", False),
        ("out_of_network_coverage", True),
        ("deductible_remaining", 250.0),
        ("copay", 30.0),
        ("coinsurance", 0.2),
    ],
)
async def test_record_benefit_happy_path(
    make_session: MakeSession, field: str, value: bool | float
):
    s = make_session()
    result = await tools.dispatch(
        ToolCall(name="record_benefit", args={"field": field, "value": value}), s
    )
    assert result.success
    assert result.advanced_call_state
    assert getattr(s.benefits, field) == value


async def test_record_benefit_bool_into_float_field_rejected(make_session: MakeSession):
    s = make_session()
    result = await tools.dispatch(
        ToolCall(name="record_benefit", args={"field": "deductible_remaining", "value": True}), s
    )
    assert not result.success
    assert "expected float" in result.message
    assert "got bool" in result.message
    assert s.benefits.deductible_remaining is None


async def test_record_benefit_float_into_bool_field_rejected(make_session: MakeSession):
    s = make_session()
    result = await tools.dispatch(
        ToolCall(name="record_benefit", args={"field": "active", "value": 1.0}), s
    )
    assert not result.success
    assert "expected bool" in result.message
    assert "got float" in result.message
    assert s.benefits.active is None


async def test_record_benefit_negative_float_rejected(make_session: MakeSession):
    s = make_session()
    result = await tools.dispatch(
        ToolCall(name="record_benefit", args={"field": "copay", "value": -5.0}), s
    )
    assert not result.success
    assert "non-negative" in result.message
    assert s.benefits.copay is None


async def test_record_benefit_zero_float_accepted(make_session: MakeSession):
    """Boundary of `< 0`: 0.0 is a valid copay (no copay due)."""
    s = make_session()
    result = await tools.dispatch(
        ToolCall(name="record_benefit", args={"field": "copay", "value": 0.0}), s
    )
    assert result.success
    assert s.benefits.copay == 0.0


async def test_record_benefit_none_value_is_accepted_no_op(make_session: MakeSession):
    s = make_session()
    s.benefits.copay = 30.0
    result = await tools.dispatch(
        ToolCall(name="record_benefit", args={"field": "copay", "value": None}), s
    )
    assert result.success
    assert not result.advanced_call_state  # no advancement → watchdog still ticks
    assert s.benefits.copay == 30.0  # not cleared


# --- transfer_to_rep --------------------------------------------------------


async def test_transfer_to_rep_flips_mode(make_session: MakeSession):
    s = make_session()
    assert s.mode == "ivr"
    result = await tools.dispatch(ToolCall(name="transfer_to_rep", args={}), s)
    assert result.success
    assert s.mode == "rep"
    assert result.side_effect is None


async def test_transfer_to_rep_idempotent_when_already_rep(make_session: MakeSession):
    """Calling transfer_to_rep twice doesn't break anything."""
    s = make_session(mode="rep")
    result = await tools.dispatch(ToolCall(name="transfer_to_rep", args={}), s)
    assert result.success
    assert s.mode == "rep"


# --- complete_call ----------------------------------------------------------


async def test_complete_call_sets_completion_reason_and_emits_hangup(make_session: MakeSession):
    s = make_session()
    result = await tools.dispatch(
        ToolCall(name="complete_call", args={"reason": "benefits_extracted"}), s
    )
    assert result.success
    assert s.completion_reason == "benefits_extracted"
    assert s.done
    assert isinstance(result.side_effect, HangupIntent)


async def test_complete_call_invalid_reason_rejected(make_session: MakeSession):
    s = make_session()
    result = await tools.dispatch(ToolCall(name="complete_call", args={"reason": "bogus"}), s)
    assert not result.success
    assert s.completion_reason is None
    assert not s.done


# --- fail_with_reason -------------------------------------------------------


async def test_fail_with_reason_in_ivr_mode_completes_as_llm_aborted(make_session: MakeSession):
    """Distinct from `ivr_no_progress` (the watchdog's reason) so dashboards
    can tell deliberate LLM aborts apart from watchdog timeouts."""
    s = make_session(mode="ivr")
    result = await tools.dispatch(
        ToolCall(name="fail_with_reason", args={"reason": "menu loop with no eligibility option"}),
        s,
    )
    assert result.success
    assert s.completion_reason == "llm_aborted_ivr"
    assert s.completion_note == "menu loop with no eligibility option"
    assert s.done
    assert "menu loop" in result.message
    assert isinstance(result.side_effect, HangupIntent)


async def test_fail_with_reason_in_rep_mode_completes_as_llm_aborted(make_session: MakeSession):
    s = make_session(mode="rep")
    result = await tools.dispatch(
        ToolCall(name="fail_with_reason", args={"reason": "rep refused to disclose"}),
        s,
    )
    assert result.success
    assert s.completion_reason == "llm_aborted_rep"
    assert s.completion_note == "rep refused to disclose"
    assert s.done


# --- send_dtmf purpose flag ------------------------------------------------


async def test_send_dtmf_purpose_rep_sets_session_flag(make_session: MakeSession):
    """`purpose='rep'` arms the transition window. Catches a regression where
    the dispatcher silently ignores the new arg."""
    s = make_session()
    assert not s.rep_pending
    result = await tools.dispatch(
        ToolCall(name="send_dtmf", args={"digits": "9", "purpose": "rep"}),
        s,
    )
    assert result.success
    assert result.advanced_call_state
    assert s.rep_pending


async def test_send_dtmf_purpose_rep_emits_audit_log(make_session: MakeSession):
    """Arming `rep_pending` MUST emit a structured `rep_digit_pressed`
    event so post-mortems can distinguish 'rep never arrived' from
    'wrong digit pressed'. Without this log, the only audit signal would
    be the tool_call entry in history — which can be torn by mid-turn
    cancellation. Locks the observability contract added in this PR.
    """
    s = make_session()
    s.recent_menu_options = ["1", "2", "9"]
    with structlog.testing.capture_logs() as captured:
        result = await tools.dispatch(
            ToolCall(name="send_dtmf", args={"digits": "9", "purpose": "rep"}),
            s,
        )
    assert result.success
    arming_events = [
        e
        for e in captured
        if e.get("event") == "rep_digit_pressed" and e.get("log_level") == "info"
    ]
    assert len(arming_events) == 1
    assert arming_events[0]["digits"] == "9"
    assert arming_events[0]["recent_menu_options"] == ["1", "2", "9"]


async def test_send_dtmf_default_purpose_does_not_set_rep_flag(make_session: MakeSession):
    """Default `purpose='menu'` (omitted) leaves `rep_pending=False`. Catches
    a regression where the default flips to `rep` or the field becomes
    required."""
    s = make_session()
    result = await tools.dispatch(ToolCall(name="send_dtmf", args={"digits": "1"}), s)
    assert result.success
    assert not s.rep_pending


async def test_send_dtmf_purpose_invalid_rejected(make_session: MakeSession):
    """`purpose` is a closed Literal — anything outside the {menu, rep} set
    must reject at the schema layer (no silent coercion)."""
    s = make_session()
    result = await tools.dispatch(
        ToolCall(name="send_dtmf", args={"digits": "9", "purpose": "agent"}),
        s,
    )
    assert not result.success
    assert "invalid args" in result.message
    assert not s.rep_pending


# --- wait tool: outside the transition window -----------------------------


async def test_wait_outside_transition_is_free(make_session: MakeSession):
    """Before any rep digit is pressed, `wait` is free — no timer, no
    state mutation. Models the IVR opening greeting case ('Welcome to
    Aetna') where there's no menu yet to act on."""
    s = make_session()
    result = await tools.dispatch(ToolCall(name="wait", args={}), s)
    assert result.success
    assert not result.advanced_call_state
    assert s.ivr_wait_started_at is None
    assert s.completion_reason is None


async def test_wait_after_transfer_is_free(make_session: MakeSession):
    """After `transfer_to_rep` clears `rep_pending`, subsequent waits are
    untimed. (Rep mode shouldn't emit `wait` in practice, but the
    dispatcher path must not arm a stale hold timer.)"""
    s = make_session()
    s.rep_pending = True
    await tools.dispatch(ToolCall(name="transfer_to_rep", args={}), s)
    assert not s.rep_pending  # transfer cleared the flag
    result = await tools.dispatch(ToolCall(name="wait", args={}), s)
    assert result.success
    assert s.ivr_wait_started_at is None


async def test_transfer_to_rep_clears_both_transition_flags(make_session: MakeSession):
    """Belt-and-suspenders: `transfer_to_rep` must explicitly clear BOTH
    `rep_pending` AND `ivr_wait_started_at`. The dispatch-entry reset
    handles the timer for the normal call path, but a future refactor
    that invokes `_dispatch_transfer_to_rep` directly (test harness, etc.)
    must not leave stale transition state.
    """
    s = make_session()
    s.rep_pending = True
    s.ivr_wait_started_at = time.monotonic()  # arbitrary non-None
    result = await tools.dispatch(ToolCall(name="transfer_to_rep", args={}), s)
    assert result.success
    assert not s.rep_pending
    assert s.ivr_wait_started_at is None


# --- wait tool: inside the transition window -----------------------------


async def test_wait_inside_transition_arms_timer(make_session: MakeSession):
    """First `wait` after `purpose='rep'` was pressed sets the clock."""
    s = make_session()
    await tools.dispatch(
        ToolCall(name="send_dtmf", args={"digits": "9", "purpose": "rep"}),
        s,
    )
    assert s.rep_pending
    assert s.ivr_wait_started_at is None  # rep_pending set, but no wait yet
    result = await tools.dispatch(ToolCall(name="wait", args={}), s)
    assert result.success
    assert s.ivr_wait_started_at is not None
    assert s.completion_reason is None


async def test_wait_inside_transition_under_budget_does_not_terminate(
    make_session: MakeSession,
):
    """Subsequent `wait` calls within budget keep the call alive."""
    s = make_session()
    s.rep_pending = True
    s.ivr_wait_started_at = time.monotonic() - 60.0  # 1 min into hold
    result = await tools.dispatch(ToolCall(name="wait", args={}), s)
    assert result.success
    assert s.completion_reason is None


async def test_wait_inside_transition_past_budget_terminates(make_session: MakeSession):
    """After 15 min of hold, the call terminates with `ivr_hold_timeout`."""
    s = make_session()
    s.rep_pending = True
    # Set the timer to 16 min ago (past the 15-min budget).
    s.ivr_wait_started_at = time.monotonic() - (16 * 60)
    result = await tools.dispatch(ToolCall(name="wait", args={}), s)
    assert result.success
    assert s.completion_reason == "ivr_hold_timeout"
    assert s.done
    assert isinstance(result.side_effect, HangupIntent)


async def test_advancing_tool_resets_hold_timer(make_session: MakeSession):
    """Any non-`wait` dispatch clears `ivr_wait_started_at`. Models the
    case where the IVR sends a NEW menu mid-hold — pressing in response
    breaks the hold and re-entering wait gets a fresh budget."""
    s = make_session()
    s.rep_pending = True
    s.ivr_wait_started_at = time.monotonic() - 10.0  # 10s into hold

    # Pressing a digit (any tool other than wait) clears the timer.
    s.recent_menu_options = ["1"]
    await tools.dispatch(ToolCall(name="send_dtmf", args={"digits": "1"}), s)
    assert s.ivr_wait_started_at is None

    # Re-entering wait state (rep_pending still True) gets a fresh start.
    result = await tools.dispatch(ToolCall(name="wait", args={}), s)
    assert result.success
    assert s.ivr_wait_started_at is not None  # freshly armed


# --- agent/telephony/dtmf.py -----------------------------------------------


async def test_send_digits_calls_twilio_with_play_twiml():
    """Twilio REST mid-stream wrapper builds the right TwiML payload."""
    fake_client = MagicMock()
    await send_digits(fake_client, "CA1234", "1#")
    fake_client.calls.assert_called_once_with("CA1234")
    fake_client.calls("CA1234").update.assert_called_with(
        twiml='<Response><Play digits="1#"/></Response>'
    )
