"""Outbound dialer with a hard allowlist check.

Every outbound call must pass through `dial()`. A typo in a Twilio number env
var should never be able to reach a real phone — the allowlist is the fence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol


class _TwilioCalls(Protocol):
    """Structural shape of the `.calls` resource on a Twilio REST client.
    Tighter than `Any` for the surface we touch; tests' mocks match structurally."""

    def create(self, *, to: str, from_: str, url: str | None) -> Any: ...


class TwilioClientLike(Protocol):
    calls: _TwilioCalls


class DestinationNotAllowedError(Exception):
    """Raised when `dial()` is called with a number outside the allowlist."""


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
        raise DestinationNotAllowedError(f"Destination {to!r} is not in ALLOWED_DESTINATIONS")


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
        raise RuntimeError("twilio_client must be provided to dial()")

    call: Any = twilio_client.calls.create(to=to, from_=from_, url=url)
    return DialResult(call_sid=str(call.sid), to=to)
