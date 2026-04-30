# pyright: strict
"""Outbound dialer with a hard allowlist check.

Every outbound call must pass through `dial()`. A typo in a Twilio number env
var should never be able to reach a real phone — the allowlist is the fence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from agent.errors import ConfigurationError

# Re-exported so `from agent.telephony.dialer import DestinationNotAllowedError`
# continues to resolve; the canonical definition lives in `agent.errors`.
from agent.errors import DestinationNotAllowedError as DestinationNotAllowedError


class _TwilioCallInstance(Protocol):
    """The fields we read on a placed Twilio call. Restricted to `sid` —
    expanding this to mirror Twilio's full `CallInstance` would couple us to
    SDK internals; pyright will fail if we ever read more without updating
    here."""

    sid: str


class _TwilioCalls(Protocol):
    """Structural shape of the `.calls` resource on a Twilio REST client.
    Test mocks satisfy this without inheriting any Twilio types."""

    def create(self, *, to: str, from_: str, url: str | None) -> _TwilioCallInstance: ...


class TwilioClientLike(Protocol):
    calls: _TwilioCalls


@dataclass(frozen=True)
class DialResult:
    call_sid: str
    to: str


def _load_allowlist(raw: str | None) -> frozenset[str]:
    if not raw:
        return frozenset()
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def check_destination(to: str, allowlist: frozenset[str]) -> None:
    if to not in allowlist:
        raise DestinationNotAllowedError(
            f"Destination {to!r} is not in ALLOWED_DESTINATIONS",
            destination=to,
        )


def dial(
    to: str,
    from_: str,
    *,
    twilio_client: TwilioClientLike | None = None,
    url: str | None = None,
    allowlist: frozenset[str] | None = None,
) -> DialResult:
    """Place an outbound call, gated by the allowlist.

    `twilio_client` is injected so unit tests can pass a fake. `url` is the
    TwiML webhook URL Twilio will fetch once the call connects.
    """
    effective_allowlist = (
        allowlist
        if allowlist is not None
        else _load_allowlist(os.environ.get("ALLOWED_DESTINATIONS"))
    )
    check_destination(to, effective_allowlist)

    if twilio_client is None:
        raise ConfigurationError(
            "twilio_client must be provided to dial()", setting="twilio_client"
        )

    call = twilio_client.calls.create(to=to, from_=from_, url=url)
    return DialResult(call_sid=call.sid, to=to)
