# pyright: strict
"""Unit tests for the schemas — Benefits, tool args, CallSession."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from agent.schemas import (
    Benefits,
    CallSession,
    CompleteCallArgs,
    DTMFIntent,
    FailWithReasonArgs,
    HangupIntent,
    PatientInfo,
    RecordBenefitArgs,
    RepTurnOutput,
    SendDTMFArgs,
    SpeakArgs,
    SpeakIntent,
    ToolCall,
    ToolResult,
    TransferToRepArgs,
    Turn,
)

# --- Benefits relaxation ----------------------------------------------------


def test_benefits_default_construction_has_all_none():
    """Required for empty CallSession.benefits — partial extraction starts blank."""
    b = Benefits()
    assert b.active is None
    assert b.deductible_remaining is None
    assert b.copay is None
    assert b.coinsurance is None
    assert b.out_of_network_coverage is None


def test_benefits_partial_construction_works():
    b = Benefits(active=True, copay=30.0)
    assert b.active is True
    assert b.copay == 30.0
    assert b.deductible_remaining is None


def test_benefits_full_construction_still_works():
    b = Benefits(
        active=True,
        deductible_remaining=250.0,
        copay=30.0,
        coinsurance=0.2,
        out_of_network_coverage=False,
    )
    assert b.active is True
    assert b.deductible_remaining == 250.0


# --- Tool argument schemas ---------------------------------------------------


def test_send_dtmf_args_accepts_digits():
    SendDTMFArgs(digits="1")
    SendDTMFArgs(digits="123#")


def test_speak_args_accepts_text():
    SpeakArgs(text="Hello.")


@pytest.mark.parametrize(
    ("model", "field_name", "bad_value"),
    [
        (SendDTMFArgs, "digits", ""),
        (SendDTMFArgs, "digits", "1" * 21),
        (SpeakArgs, "text", ""),
        (SpeakArgs, "text", "x" * 201),
        (FailWithReasonArgs, "reason", ""),
        (FailWithReasonArgs, "reason", "x" * 121),
    ],
)
def test_string_length_bounds_rejected(model: type[BaseModel], field_name: str, bad_value: str):
    """Smoke-test that min_length / max_length declarations are wired up."""
    with pytest.raises(ValidationError):
        model(**{field_name: bad_value})


def test_record_benefit_args_field_must_be_valid():
    RecordBenefitArgs(field="active", value=True)
    RecordBenefitArgs(field="deductible_remaining", value=250.0)
    RecordBenefitArgs(field="active", value=None)
    with pytest.raises(ValidationError):
        RecordBenefitArgs.model_validate({"field": "not_a_field", "value": True})


@pytest.mark.parametrize(
    ("value", "expected_type"),
    [
        (True, bool),
        (False, bool),
        (250.0, float),
        (0.0, float),
        (None, type(None)),
    ],
)
def test_record_benefit_args_preserves_bool_vs_float(
    value: bool | float | None, expected_type: type
):
    """Pydantic v2 smart mode used to coerce False↔0.0 silently. Lock the
    distinction in — the tool dispatcher uses `value`'s type to validate
    against the field's expected type (`bool` for active/oon, `float` for
    deductible/copay/coinsurance)."""
    args = RecordBenefitArgs(field="active", value=value)
    assert type(args.value) is expected_type


def test_transfer_to_rep_args_takes_no_args():
    args = TransferToRepArgs()
    assert args.model_dump() == {}


def test_complete_call_args_reason_must_be_valid():
    CompleteCallArgs(reason="benefits_extracted")
    CompleteCallArgs(reason="ivr_dead_end")
    CompleteCallArgs(reason="user_hangup")
    with pytest.raises(ValidationError):
        CompleteCallArgs.model_validate({"reason": "any_string"})


def test_fail_with_reason_args_accepts_reason():
    FailWithReasonArgs(reason="watchdog tripped")


# --- ToolCall / ToolResult / SideEffectIntent --------------------------------


def test_tool_call_constructs():
    call = ToolCall(name="send_dtmf", args={"digits": "1"}, id="toolu_01")
    assert call.name == "send_dtmf"
    assert call.args == {"digits": "1"}
    assert call.id == "toolu_01"


def test_tool_call_id_optional():
    call = ToolCall(name="transfer_to_rep", args={})
    assert call.id is None


def test_tool_call_unknown_name_rejected():
    with pytest.raises(ValidationError):
        ToolCall.model_validate({"name": "bogus_tool", "args": {}})


def test_tool_result_with_dtmf_side_effect():
    result = ToolResult(
        success=True,
        advanced_call_state=True,
        message="DTMF 1 dispatched.",
        side_effect=DTMFIntent(digits="1"),
    )
    assert isinstance(result.side_effect, DTMFIntent)
    assert result.side_effect.digits == "1"


def test_tool_result_with_speak_side_effect():
    result = ToolResult(
        success=True,
        advanced_call_state=True,
        message="Speaking.",
        side_effect=SpeakIntent(text="Hello."),
    )
    assert isinstance(result.side_effect, SpeakIntent)


def test_tool_result_with_hangup_side_effect():
    result = ToolResult(
        success=True,
        advanced_call_state=True,
        message="Hanging up.",
        side_effect=HangupIntent(),
    )
    assert isinstance(result.side_effect, HangupIntent)


def test_tool_result_validation_failure_path():
    """`advanced_call_state=False` is the watchdog signal for rejected args."""
    result = ToolResult(
        success=False,
        advanced_call_state=False,
        message="digit 9 not in recent menu options [1, 2, 3]; pick again",
        side_effect=None,
    )
    assert not result.success
    assert not result.advanced_call_state
    assert result.side_effect is None


@pytest.mark.parametrize(
    ("kind", "extra", "expected_type"),
    [
        ("dtmf", {"digits": "1"}, DTMFIntent),
        ("speak", {"text": "Hello."}, SpeakIntent),
        ("hangup", {}, HangupIntent),
    ],
)
def test_side_effect_intent_discriminator_round_trip(
    kind: str, extra: dict[str, object], expected_type: type
):
    """Pydantic must dispatch on `kind` when rehydrating from a dict (Langfuse
    trace replay, JSON fixtures). Without the explicit discriminator,
    smart-mode could silently coerce in ambiguous cases."""
    payload = {
        "success": True,
        "advanced_call_state": True,
        "message": "ok",
        "side_effect": {"kind": kind, **extra},
    }
    rehydrated = ToolResult.model_validate(payload)
    assert isinstance(rehydrated.side_effect, expected_type)


# --- RepTurnOutput -----------------------------------------------------------


def test_rep_turn_output_extracting_phase_with_partial_benefits():
    out = RepTurnOutput(
        reply="Got it. What's the deductible remaining?",
        extracted=Benefits(active=True),
        phase="extracting",
    )
    assert out.phase == "extracting"
    assert out.extracted.active is True
    assert out.extracted.deductible_remaining is None
    assert out.reasoning is None


def test_rep_turn_output_complete_phase_full_extract(benefits: Benefits):
    out = RepTurnOutput(
        reply="That's everything I needed, thanks Sam, have a great day.",
        extracted=benefits,
        phase="complete",
        reasoning="all five fields filled with high confidence.",
    )
    assert out.phase == "complete"


def test_rep_turn_output_silent_reply_with_empty_extraction():
    """Hold-music announcement case — stay silent, extract nothing."""
    out = RepTurnOutput(reply="", extracted=Benefits(), phase="extracting")
    assert out.reply == ""
    assert all(getattr(out.extracted, f) is None for f in Benefits.model_fields)


def test_rep_turn_output_unknown_phase_rejected():
    # Use `model_validate` so a runtime-only test of validation logic doesn't
    # depend on bypassing the static type check.
    with pytest.raises(ValidationError):
        RepTurnOutput.model_validate(
            {"reply": "x", "extracted": Benefits().model_dump(), "phase": "dunno"}
        )


# --- Turn --------------------------------------------------------------------


def test_turn_user():
    t = Turn(role="user", content="Press 1 for eligibility")
    assert t.role == "user"
    assert t.tool_call is None
    assert t.tool_result is None
    assert t.extracted is None


def test_turn_tool_call():
    t = Turn(
        role="tool_call",
        tool_call=ToolCall(name="send_dtmf", args={"digits": "1"}),
    )
    assert t.role == "tool_call"
    assert t.tool_call is not None
    assert t.tool_call.name == "send_dtmf"


def test_turn_assistant_with_extraction():
    t = Turn(
        role="assistant",
        content="Got it.",
        extracted=Benefits(copay=30.0),
    )
    assert t.extracted is not None
    assert t.extracted.copay == 30.0


# --- CallSession -------------------------------------------------------------


def test_call_session_defaults(patient: PatientInfo):
    s = CallSession(call_sid="CA-test", patient=patient)
    assert s.mode == "ivr"
    assert s.history == []
    assert isinstance(s.benefits, Benefits)
    assert s.benefits.active is None  # relies on Benefits relaxation
    assert s.recent_menu_options == []
    assert s.turn_count == 0
    assert s.ivr_no_progress_turns == 0
    assert s.stuck_turns == 0
    assert s.done is False
    assert s.completion_reason is None
    assert s.completion_note is None


def test_call_session_history_is_per_instance(patient: PatientInfo):
    """Default-factory'd lists must not share state between sessions."""
    s1 = CallSession(call_sid="CA-1", patient=patient)
    s2 = CallSession(call_sid="CA-2", patient=patient)
    s1.history.append(Turn(role="user", content="hi"))
    assert s2.history == []


def test_call_session_done_derives_from_completion_reason(patient: PatientInfo):
    """`done` is a property — single source of truth is `completion_reason`."""
    s = CallSession(call_sid="CA-d", patient=patient)
    assert s.done is False
    s.completion_reason = "rep_complete"
    assert s.done is True
