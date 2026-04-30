# pyright: strict
"""Error taxonomy for the agent.

`AgentError` is the base for every error the agent itself raises. Catch
`AgentError` to handle any agent-level failure without sweeping in
infrastructure errors (network, OS, asyncio, third-party SDKs).

Subclasses exist so callers can `except SpecificError:` instead of
string-matching on `RuntimeError` messages — string matches rot when an
error message changes; type matches do not.

The dispatcher's *validation failures* (bad arg shapes, missing menu
options) deliberately do NOT raise — they return a `ToolResult` whose
message goes back into history so the LLM can re-pick. Errors here are
for cases the LLM cannot recover from on its own.
"""

from __future__ import annotations

from typing import Literal

from agent.schemas import IntentKind, ToolName

# Anthropic-style stop reasons surfaced via Claude's `messages.parse()`.
# `unknown` is the escape hatch for the `getattr(..., "stop_reason", "unknown")`
# fallback in the rep client — and for any future provider whose stop_reason
# set we don't recognise. Pinning the literal keeps log/dashboard aggregations
# clean instead of grouping on typos.
LLMStopReason = Literal[
    "end_turn",
    "max_tokens",
    "stop_sequence",
    "tool_use",
    "refusal",
    "unknown",
]


class AgentError(Exception):
    """Base for every agent-internal error."""


class ConfigurationError(AgentError):
    """Missing or invalid wiring — env vars, dependency injection, etc.

    Raised at construction time so a misconfigured deployment dies before
    audio starts flowing, not several seconds into a live call. `setting`
    names the offending env var or DI parameter so operators don't have
    to grep the message.
    """

    def __init__(self, message: str, *, setting: str) -> None:
        super().__init__(message)
        self.setting = setting


class LLMRefusalError(AgentError):
    """The LLM declined to answer or returned unparseable output.

    Carries `stop_reason` and `response_id` so dashboards can correlate
    the failure to the underlying provider trace. Callers may choose to
    retry with a different prompt or surface the refusal as a
    `completion_reason`.
    """

    def __init__(
        self,
        message: str,
        *,
        stop_reason: LLMStopReason,
        response_id: str,
    ) -> None:
        super().__init__(message)
        self.stop_reason = stop_reason
        self.response_id = response_id


class ToolDispatchError(AgentError):
    """An exception escaped the tool dispatcher's handler.

    Validation rejections are NOT this — they return a `ToolResult`. This
    is for unhandled exceptions in the dispatch handlers, which the
    runner converts to a structured completion reason. `tool_name`
    attributes the failure without parsing the message; `None` is
    accepted for the rare malformed-frame case where the tool name
    couldn't be resolved.
    """

    def __init__(self, message: str, *, tool_name: ToolName | None = None) -> None:
        super().__init__(message)
        self.tool_name = tool_name


class ActuatorError(AgentError):
    """The actuator could not execute a side-effect intent.

    `intent_kind` (typed against the canonical `SideEffectIntent.kind`
    discriminator from `agent.schemas`) lets logs attribute the failure
    without parsing the message and fails type-check at the raise site
    if a wrong value is passed.
    """

    def __init__(self, message: str, *, intent_kind: IntentKind) -> None:
        super().__init__(message)
        self.intent_kind = intent_kind


class DestinationNotAllowedError(AgentError):
    """Outbound dial blocked by the `ALLOWED_DESTINATIONS` fence.

    The fence is non-negotiable: a typo in a Twilio env var must never
    be able to reach a real phone. `destination` is the offending
    number, surfaced as a structured field so log aggregators can group
    blocked-dial events without parsing the message.
    """

    def __init__(self, message: str, *, destination: str) -> None:
        super().__init__(message)
        self.destination = destination
