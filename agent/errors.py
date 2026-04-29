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

# Mirrors the `kind` discriminator on `SideEffectIntent` in `agent.schemas`.
# Defined here as a Literal alias rather than imported to keep the error
# module standalone — `agent.schemas` will import from `agent.errors` once
# the M8'/E2 migration lands, and the cycle is easier avoided than fixed.
IntentKind = Literal["speak", "dtmf", "hangup"]


class AgentError(Exception):
    """Base for every agent-internal error."""


class ConfigurationError(AgentError):
    """Missing or invalid wiring — env vars, dependency injection, etc.

    Raised at construction time so a misconfigured deployment dies before
    audio starts flowing, not several seconds into a live call.
    """


class LLMRefusalError(AgentError):
    """The LLM declined to answer or returned unparseable output.

    Carries `stop_reason` and `response_id` so dashboards (Langfuse, log
    aggregators) can correlate the failure to the underlying provider
    trace. Callers may choose to retry with a different prompt or
    surface the refusal as a `completion_reason`.
    """

    def __init__(self, message: str, *, stop_reason: str, response_id: str) -> None:
        super().__init__(message)
        self.stop_reason = stop_reason
        self.response_id = response_id


class ToolDispatchError(AgentError):
    """An exception escaped the tool dispatcher's handler.

    Validation rejections are NOT this — they return a `ToolResult`.
    This is for unhandled exceptions in `_dispatch_*` helpers, which the
    runner converts to a `tool_dispatch_exception` completion reason.
    `tool_name` lets the runner attribute the failure without parsing the
    message; `None` is acceptable for cases where the tool name itself
    couldn't be resolved (e.g. malformed dispatch frame).
    """

    def __init__(self, message: str, *, tool_name: str | None = None) -> None:
        super().__init__(message)
        self.tool_name = tool_name


class ActuatorError(AgentError):
    """The actuator could not execute a side-effect intent.

    `intent_kind` identifies which `SideEffectIntent` variant failed so
    logs can attribute the failure without parsing the message. Typed as
    a Literal so a wrong value fails type-check at the raise site.
    """

    def __init__(self, message: str, *, intent_kind: IntentKind) -> None:
        super().__init__(message)
        self.intent_kind = intent_kind


class DestinationNotAllowedError(AgentError):
    """Outbound dial blocked by the `ALLOWED_DESTINATIONS` fence.

    The fence is non-negotiable: a typo in a Twilio env var must never
    be able to reach a real phone. This error surfaces only when the
    target number is genuinely outside the allowlist.
    """
