"""Executes side-effect intents emitted by the tool dispatcher.

The tools layer is a pure function — it returns `ToolResult` plus an optional
`SideEffectIntent`. The actuator turns those intents into real side effects:

- `SpeakIntent` → push text into `session.out_queue`. The `state_processor`
  Pipecat adapter pumps the queue into `TextFrame`s for Cartesia. This avoids
  coupling the actuator to the FrameProcessor.
- `DTMFIntent` → call Twilio REST `<Play digits>` via `agent.telephony.dtmf`.
  Requires a Twilio client and the live `call_sid`.
- `HangupIntent` → no immediate I/O; the runner observes
  `session.completion_reason` (set by `complete_call` / `fail_with_reason` in
  the dispatcher) and exits the consume loop. The intent is still returned
  for completeness — and so a future Twilio-side hangup can be wired here
  without changing the dispatcher.
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

from agent.schemas import (
    CallSession,
    DTMFIntent,
    HangupIntent,
    SideEffectIntent,
    SpeakIntent,
)
from agent.telephony.dtmf import send_digits


class Actuator(Protocol):
    async def execute(self, intent: SideEffectIntent) -> None: ...


class CallActuator:
    """Per-call actuator. One per `CallSessionRunner`.

    `twilio_client` is optional — offline unit tests omit it and any
    `DTMFIntent` raises. Live calls inject the real `twilio.rest.Client`.
    """

    def __init__(
        self,
        session: CallSession,
        out_queue: asyncio.Queue[str],
        twilio_client: Any | None = None,
    ) -> None:
        self.session = session
        self.out_queue = out_queue
        self.twilio_client = twilio_client

    async def execute(self, intent: SideEffectIntent) -> None:
        if isinstance(intent, SpeakIntent):
            # `out_queue` is bounded with backpressure — TTS shouldn't fall
            # behind, and if it does we want the LLM to feel that pressure
            # rather than dropping spoken text on the floor.
            await self.out_queue.put(intent.text)
            return
        if isinstance(intent, DTMFIntent):
            if self.twilio_client is None:
                raise RuntimeError(
                    "DTMFIntent emitted but actuator has no twilio_client; "
                    "wire one in via CallSessionRunner construction."
                )
            await send_digits(self.twilio_client, self.session.call_sid, intent.digits)
            return
        if isinstance(intent, HangupIntent):
            # Termination is driven by `session.completion_reason`, which the
            # dispatcher already set when the LLM emitted complete_call /
            # fail_with_reason. Nothing to do here on the agent side; the
            # runner will exit its consume loop on the next iteration.
            return
        raise RuntimeError(f"actuator received unknown intent: {intent!r}")
