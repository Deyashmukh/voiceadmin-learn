# pyright: strict
"""IVR tool registry + dispatcher.

The LLM emits tool calls; the dispatcher validates each one structurally
(via the per-tool Pydantic arg model) and contextually (against `CallSession`
state — recent menu options, benefit field types, etc.) before mutating the
session or returning a side-effect intent for the actuator to execute.

Validation failures are NOT exceptions — they return a `ToolResult` whose
message goes back into history, so the LLM can re-pick on the next turn.
The watchdog uses `advanced_call_state` to distinguish a successful action
from a rejection round-trip.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, ValidationError

from agent.logging_config import log
from agent.observability import observe, set_current_span_name
from agent.schemas import (
    BenefitField,
    CallSession,
    CompleteCallArgs,
    DTMFIntent,
    FailWithReasonArgs,
    HangupIntent,
    RecordBenefitArgs,
    SendDTMFArgs,
    SpeakArgs,
    SpeakIntent,
    ToolCall,
    ToolName,
    ToolResult,
    TransferToRepArgs,
    WaitArgs,
)

# Per-tool argument schemas. Adding a new tool is a one-line registry update +
# the matching handler branch in `dispatch` below.
TOOL_ARG_MODELS: dict[ToolName, type[BaseModel]] = {
    "send_dtmf": SendDTMFArgs,
    "speak": SpeakArgs,
    "record_benefit": RecordBenefitArgs,
    "transfer_to_rep": TransferToRepArgs,
    "complete_call": CompleteCallArgs,
    "fail_with_reason": FailWithReasonArgs,
    "wait": WaitArgs,
}

# Descriptions surfaced to the IVR LLM via the Groq tool schema. Each one
# tells the LLM when to pick the tool, not what its args are (the JSON
# schema covers args). Keep terse — long descriptions waste tokens.
_TOOL_DESCRIPTIONS: dict[ToolName, str] = {
    "send_dtmf": ("Press DTMF tones on the live phone call. Use to navigate IVR menus."),
    "speak": (
        "Speak the LITERAL text aloud over TTS. In IVR mode this is ONLY "
        "for providing the patient's identifiers (name, member ID, DOB) "
        "when the IVR explicitly prompts for them, OR for the spoken "
        "repeat command an IVR menu offered (e.g., 'repeat'). NEVER use "
        "to greet, converse, narrate, or acknowledge — the IVR doesn't "
        "respond to chit-chat and it wastes turns."
    ),
    "record_benefit": (
        "Record a benefit field value the IVR (or rep) read aloud. "
        "Examples: copay=30.0, active=true."
    ),
    "transfer_to_rep": (
        "Mark that the call has been handed off to a human rep. After this, "
        "the rep-mode LLM takes over. One-way; cannot transfer back to IVR."
    ),
    "complete_call": (
        "End the call cleanly with a structured reason "
        "(benefits_extracted | ivr_dead_end | user_hangup)."
    ),
    "fail_with_reason": (
        "Abort the call with a free-form reason. Use only when no tool fits "
        "and the call cannot continue."
    ),
    "wait": (
        "Acknowledge the current utterance without acting. Use for IVR "
        "opening greetings ('Welcome to Aetna' BEFORE any menu has appeared "
        "— this is the IVR itself, not a rep), hold music, hold announcements "
        "('please continue to hold'), filler ('calls may be recorded'), and "
        "any non-actionable input. After `send_dtmf(purpose='rep')`, repeated "
        "`wait` calls are bounded by a 15-min hold budget. Before any rep "
        "digit is pressed, NEVER use `transfer_to_rep` for greetings — they "
        "are the IVR opening, not a rep arriving."
    ),
}


def groq_tool_schemas() -> list[dict[str, object]]:
    """Build Groq/OpenAI-style function definitions from `TOOL_ARG_MODELS`.

    Each entry surfaces to the IVR LLM as a callable tool with a
    Pydantic-derived JSON schema for its args. Wired into the runner
    via `agent.main`'s `groq_tool_schemas()` argument.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": _TOOL_DESCRIPTIONS[name],
                "parameters": model.model_json_schema(),
            },
        }
        for name, model in TOOL_ARG_MODELS.items()
    ]


# `record_benefit` validates that the value type matches the field's expected
# type. Pydantic's smart-mode would coerce `False` ↔ `0.0` silently; the test
# suite locks the distinction in.
_BENEFIT_FIELD_TYPES: dict[BenefitField, type] = {
    "active": bool,
    "deductible_remaining": float,
    "copay": float,
    "coinsurance": float,
    "out_of_network_coverage": bool,
}

# DTMF keys always allowed regardless of the most recent menu options.
_UNIVERSAL_DTMF_KEYS = frozenset("#*")

# Wall-clock budget for the transition phase between the LLM emitting
# `send_dtmf(purpose="rep")` and `transfer_to_rep`. Real eligibility-
# verification holds run 5-10min on average; 15min covers the long tail
# without letting a stuck call linger indefinitely. Bounded only across a
# single hold period — any non-`wait` tool call clears the timer.
_HOLD_BUDGET_S = 15 * 60


def _reject(message: str) -> ToolResult:
    """Validation-failure result. The message goes back into history so the
    LLM can self-correct on the next turn."""
    return ToolResult(success=False, advanced_call_state=False, message=message)


@observe(name="tool_dispatch")
async def dispatch(call: ToolCall, session: CallSession) -> ToolResult:
    """Validate `call` against its schema + session state, then apply effects.

    Returns a `ToolResult` describing what happened. Side effects that the
    actuator must execute (DTMF, TTS, hangup) are returned via
    `ToolResult.side_effect`; in-process state mutations (mode flip,
    completion reason, benefits merge) happen here directly on `session`.
    """
    # `call.name` is `ToolName` Literal — registry has every variant by
    # construction, so the lookup is total.
    set_current_span_name(f"tool_dispatch.{call.name}")
    schema = TOOL_ARG_MODELS[call.name]
    try:
        args = schema.model_validate(call.args)
    except ValidationError as exc:
        return _reject(f"invalid args for {call.name}: {exc}")

    # `wait` keeps the hold timer running across consecutive waits; every
    # other dispatch path resets it (advancing breaks the hold, even if the
    # call ultimately re-enters wait state later — that re-entry gets a
    # fresh budget). Centralized here so each non-wait dispatcher doesn't
    # need to remember to clear the timer.
    if call.name != "wait":
        session.ivr_wait_started_at = None

    match call.name:
        case "send_dtmf":
            assert isinstance(args, SendDTMFArgs)
            return _dispatch_send_dtmf(args, session)
        case "speak":
            assert isinstance(args, SpeakArgs)
            return _dispatch_speak(args)
        case "record_benefit":
            assert isinstance(args, RecordBenefitArgs)
            return _dispatch_record_benefit(args, session)
        case "transfer_to_rep":
            assert isinstance(args, TransferToRepArgs)
            return _dispatch_transfer_to_rep(session)
        case "complete_call":
            assert isinstance(args, CompleteCallArgs)
            return _dispatch_complete_call(args, session)
        case "fail_with_reason":
            assert isinstance(args, FailWithReasonArgs)
            return _dispatch_fail_with_reason(args, session)
        case "wait":
            assert isinstance(args, WaitArgs)
            return _dispatch_wait(session)


def _dispatch_send_dtmf(args: SendDTMFArgs, session: CallSession) -> ToolResult:
    if session.recent_menu_options:
        for digit in args.digits:
            if digit in _UNIVERSAL_DTMF_KEYS:
                continue
            if digit not in session.recent_menu_options:
                return _reject(
                    f"digit {digit!r} not offered by the most recent menu "
                    f"(options: {session.recent_menu_options}); pick again or "
                    "send only universal keys (#, *)."
                )
    if args.purpose == "rep":
        session.rep_pending = True
    return ToolResult(
        success=True,
        advanced_call_state=True,
        message=f"DTMF {args.digits} dispatched.",
        side_effect=DTMFIntent(digits=args.digits),
    )


def _dispatch_speak(args: SpeakArgs) -> ToolResult:
    return ToolResult(
        success=True,
        advanced_call_state=True,
        message=f"Speaking: {args.text!r}",
        side_effect=SpeakIntent(text=args.text),
    )


def _dispatch_record_benefit(args: RecordBenefitArgs, session: CallSession) -> ToolResult:
    expected_type = _BENEFIT_FIELD_TYPES[args.field]
    if args.value is None:
        # Recording None is a no-op — accepted so the LLM can clear a field it
        # later determines was wrong. No mutation, no state advancement.
        return ToolResult(
            success=True,
            advanced_call_state=False,
            message=f"record_benefit({args.field!r}) called with None; ignored.",
        )
    # bool is a subtype of int in Python, but not of float. Plain isinstance
    # would accept True for a float field. Check the bool tag explicitly.
    is_bool_value = isinstance(args.value, bool)
    if expected_type is bool and not is_bool_value:
        return _reject(_type_mismatch_msg(args.field, expected_type, args.value))
    if expected_type is float and (is_bool_value or not isinstance(args.value, float)):
        return _reject(_type_mismatch_msg(args.field, expected_type, args.value))
    if expected_type is float and args.value < 0:
        return _reject(
            f"record_benefit({args.field!r}) got negative value {args.value!r}; "
            "Benefits fields must be non-negative."
        )
    setattr(session.benefits, args.field, args.value)
    return ToolResult(
        success=True,
        advanced_call_state=True,
        message=f"recorded {args.field}={args.value}",
    )


def _type_mismatch_msg(field: str, expected_type: type, value: object) -> str:
    return (
        f"record_benefit({field!r}) expected {expected_type.__name__}, "
        f"got {type(value).__name__}; pick again."
    )


def _dispatch_transfer_to_rep(session: CallSession) -> ToolResult:
    session.mode = "rep"
    # Record the history offset at which the rep conversation begins so the
    # rep LLM doesn't receive IVR-phase user transcripts (which would arrive
    # as consecutive same-role messages — Anthropic 400s on those).
    if session.rep_mode_index is None:
        session.rep_mode_index = len(session.history)
    # Transition phase is over. Clear `rep_pending` so any stray `wait`
    # call after the flip doesn't re-arm the hold timer.
    session.rep_pending = False
    return ToolResult(
        success=True,
        advanced_call_state=True,
        message="Mode flipped to rep. Next turn routes to the rep LLM.",
    )


def _dispatch_complete_call(args: CompleteCallArgs, session: CallSession) -> ToolResult:
    session.completion_reason = args.reason
    return ToolResult(
        success=True,
        advanced_call_state=True,
        message=f"Call complete: {args.reason}.",
        side_effect=HangupIntent(),
    )


def _dispatch_fail_with_reason(args: FailWithReasonArgs, session: CallSession) -> ToolResult:
    # Use mode-specific `llm_aborted_*` reasons rather than the watchdog's
    # `ivr_no_progress` / `rep_stuck` so dashboards can tell deliberate LLM
    # aborts apart from watchdog timeouts. The free-form reason text goes on
    # `completion_note` for logs and Langfuse traces.
    session.completion_reason = "llm_aborted_ivr" if session.mode == "ivr" else "llm_aborted_rep"
    session.completion_note = args.reason
    return ToolResult(
        success=True,
        advanced_call_state=True,
        message=f"LLM aborted: {args.reason}",
        side_effect=HangupIntent(),
    )


def _dispatch_wait(session: CallSession) -> ToolResult:
    """Acknowledge a non-actionable utterance without doing anything.

    Two regimes, gated by `rep_pending`:
    - **Outside the transition window** (no rep digit pressed yet, OR already
      transferred to rep): `wait` is free. No timer, no watchdog impact.
      The IVR LLM uses this for opening greetings, mid-menu filler, etc.
    - **Inside the transition window** (`send_dtmf(purpose="rep")` was
      called and we haven't transferred yet): the first `wait` arms the
      hold timer. Each subsequent `wait` checks elapsed time against
      `_HOLD_BUDGET_S`; past that, the call terminates with
      `ivr_hold_timeout`. Any non-`wait` dispatch resets the timer (handled
      in `dispatch` above), so a successful menu interaction mid-hold
      restarts the budget.

    `advanced_call_state=False` here, but the IVR turn loop in
    `agent/call_session.py:_ivr_turn` exempts wait-only turns from the
    no-progress watchdog (`only_wait = all(c.name == "wait" ...)`). So
    a deliberate non-action doesn't get conflated with a stuck spin.
    """
    if not session.rep_pending:
        return ToolResult(
            success=True,
            advanced_call_state=False,
            message="Waiting (no rep transition pending).",
        )
    now = time.monotonic()
    if session.ivr_wait_started_at is None:
        session.ivr_wait_started_at = now
        return ToolResult(
            success=True,
            advanced_call_state=False,
            message="Waiting for rep to arrive (15min budget started).",
        )
    elapsed = now - session.ivr_wait_started_at
    if elapsed > _HOLD_BUDGET_S:
        session.completion_reason = "ivr_hold_timeout"
        log.info("ivr_hold_timeout_watchdog_tripped", elapsed_s=round(elapsed, 1))
        return ToolResult(
            success=True,
            advanced_call_state=False,
            message=f"Hold timeout after {elapsed:.0f}s; rep never arrived.",
            side_effect=HangupIntent(),
        )
    return ToolResult(
        success=True,
        advanced_call_state=False,
        message=f"Waiting for rep to arrive ({elapsed:.0f}s elapsed of {_HOLD_BUDGET_S}s).",
    )
