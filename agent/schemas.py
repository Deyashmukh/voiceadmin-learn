"""Shared data models for the voice agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Literal

from pydantic import BaseModel, Field

FALLBACK_RESPONSE = "Sorry, could you repeat that?"


class PatientInfo(BaseModel):
    """Caller/subscriber info we authenticate with on the payer call."""

    member_id: str
    first_name: str
    last_name: str
    dob: str  # ISO date string; not a date() because TwiML-friendly


class Benefits(BaseModel):
    """The five fields the call exists to extract.

    All optional so the same model represents partial extractions during a call
    (rep mode merges non-None fields turn by turn) and the final result.
    """

    active: bool | None = None
    deductible_remaining: float | None = None
    copay: float | None = None
    coinsurance: float | None = None
    out_of_network_coverage: bool | None = None


# --- Tool argument schemas ----------------------------------------------------
# Input shapes the IVR LLM emits via tool calling. The dispatcher validates
# additional invariants (digit-in-recent-menu-options, value matches field
# type) on top of Pydantic's structural validation — see agent/tools.py.

BenefitField = Literal[
    "active", "deductible_remaining", "copay", "coinsurance", "out_of_network_coverage"
]
# The reasons the LLM is allowed to pass to `complete_call`. A strict subset of
# `CompletionReason` below (which also covers dispatcher- and watchdog-driven
# terminations the LLM never names).
CompleteCallReason = Literal["benefits_extracted", "ivr_dead_end", "user_hangup"]


class SendDTMFArgs(BaseModel):
    # Restrict to the DTMF wire alphabet — these are the only tones the
    # carrier recognizes regardless of how we render them. Bounds to a real
    # menu's worth of input (member ID + #/*); 20 is generous.
    digits: str = Field(min_length=1, max_length=20, pattern=r"^[0-9*#]+$")


class SpeakArgs(BaseModel):
    text: str = Field(min_length=1, max_length=200)


class RecordBenefitArgs(BaseModel):
    field: BenefitField
    value: bool | float | None


class TransferToRepArgs(BaseModel):
    pass


class CompleteCallArgs(BaseModel):
    reason: CompleteCallReason


class FailWithReasonArgs(BaseModel):
    reason: str = Field(min_length=1, max_length=120)


# --- Tool dispatch + side effects --------------------------------------------

ToolName = Literal[
    "send_dtmf", "speak", "record_benefit", "transfer_to_rep", "complete_call", "fail_with_reason"
]

# Discriminator union for `SideEffectIntent.kind`. Named so consumers like
# `agent.errors.ActuatorError.intent_kind` can reference it without
# redeclaring — keeps the literal set canonical and drift-proof.
IntentKind = Literal["dtmf", "speak", "hangup"]


class ToolCall(BaseModel):
    """A tool invocation produced by the IVR LLM."""

    name: ToolName
    args: dict[str, object]
    id: str | None = None  # provider-assigned id, used for tool_choice round-trips


class DTMFIntent(BaseModel):
    kind: Literal["dtmf"] = "dtmf"
    digits: str


class SpeakIntent(BaseModel):
    kind: Literal["speak"] = "speak"
    text: str


class HangupIntent(BaseModel):
    kind: Literal["hangup"] = "hangup"


# Tagged union — `Field(discriminator="kind")` makes Pydantic dispatch on the
# `kind` Literal when rehydrating from a dict (Langfuse trace replay, fixtures,
# JSON logs). Without it, smart-mode silently coerces in ambiguous cases.
SideEffectIntent = Annotated[
    DTMFIntent | SpeakIntent | HangupIntent,
    Field(discriminator="kind"),
]


class IVRTurnResponse(BaseModel):
    """One IVR LLM round-trip's worth of output. The LLM may emit zero or more
    tool calls plus optional text (rare in IVR mode, but supported for asides
    like 'one moment' before a DTMF). Empty `tool_calls` is the watchdog's
    no-progress signal."""

    tool_calls: list[ToolCall] = Field(default_factory=list[ToolCall])
    text: str = ""


class ToolResult(BaseModel):
    """What the dispatcher returns for a single tool call.

    `advanced_call_state=False` means the call wasn't actually moved forward
    (e.g., validator rejected the args); the watchdog uses this signal.
    """

    success: bool
    advanced_call_state: bool
    message: str  # text payload appended to history so the LLM can re-pick on errors
    side_effect: SideEffectIntent | None = None


# --- Rep mode output ---------------------------------------------------------


class RepTurnOutput(BaseModel):
    """Single-turn output of the rep LLM. `extracted` is partial; only non-None
    fields are merged into `CallSession.benefits`."""

    reply: str  # what the agent says aloud ("" = stay silent)
    extracted: Benefits
    phase: Literal["extracting", "complete", "stuck"]
    reasoning: str | None = None


# --- Conversation history ----------------------------------------------------


class Turn(BaseModel):
    """One entry in the running call history.

    Flat model with optional fields rather than a discriminated union — history
    is rendered into LLM prompts as text, not consumed structurally, so type
    discipline at the boundary doesn't earn its keep.
    """

    role: Literal["user", "assistant", "tool_call", "tool_result"]
    content: str = ""
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    extracted: Benefits | None = None  # set when the assistant produced a partial extraction


# --- CallSession -------------------------------------------------------------

CallMode = Literal["ivr", "rep"]
# Composed so the LLM-emittable subset (`CompleteCallReason`) is reused, not
# duplicated. Adding a new termination reason here only requires updating one
# Literal — and dispatcher/watchdog reasons stay separate from LLM ones.
CompletionReason = (
    CompleteCallReason
    | Literal[
        "ivr_no_progress",  # watchdog: 2 IVR turns with no advancing tool call
        "llm_aborted_ivr",  # LLM emitted fail_with_reason in ivr mode (deliberate abort)
        "rep_complete",
        "rep_stuck",  # watchdog: phase=stuck for 2 consecutive turns
        "llm_aborted_rep",  # LLM emitted fail_with_reason in rep mode (deliberate abort)
        "tool_dispatch_exception",
        "dtmf_dispatch_failed",
        "asr_lost",
        "transport_closed",
        "consumer_died",  # consumer task crashed unhandled — safety-net so the pipeline doesn't hang on out_queue
    ]
)


@dataclass
class CallSession:
    """Per-call state owned by the call loop. One per call; never global.

    Mutable by design — `_run_turn` and the tool dispatcher both write here.
    Cancellation safety lives in the call loop, not in this struct.
    """

    call_sid: str
    patient: PatientInfo
    mode: CallMode = "ivr"
    # All Turn entries appended in chronological order: user transcripts,
    # assistant replies, tool_call entries, tool_result entries. Rendered into
    # LLM prompts as text rather than consumed structurally.
    history: list[Turn] = field(default_factory=list[Turn])
    benefits: Benefits = field(default_factory=Benefits)
    # Populated when a transcript looks like a menu prompt (parsed in the IVR
    # turn handler). `send_dtmf` validation rejects digits not in this list;
    # universal #/* are always allowed.
    recent_menu_options: list[str] = field(default_factory=list[str])
    # Distinct from `len(history)` because history mixes user/assistant/tool
    # entries. Counts completed turns from the per-turn loop's perspective:
    # one increment per `_run_turn` invocation that didn't get cancelled.
    turn_count: int = 0
    ivr_no_progress_turns: int = 0
    stuck_turns: int = 0
    completion_reason: CompletionReason | None = None
    # Free-form narrative for `fail_with_reason` and similar paths where the
    # structural `completion_reason` Literal can't capture the LLM's text.
    # Surfaced in logs and Langfuse traces; not consumed by control flow.
    completion_note: str | None = None
    # Set by `transfer_to_rep` to `len(history)` at the moment of flip, so the
    # rep LLM only sees post-flip history. Without this, the IVR phase's user
    # transcripts would arrive at the rep LLM as consecutive user messages —
    # Anthropic's API rejects messages that don't start with a user role or
    # have consecutive same-role entries.
    rep_mode_index: int | None = None

    @property
    def done(self) -> bool:
        """Single source of truth: the call is over once a reason is set."""
        return self.completion_reason is not None
