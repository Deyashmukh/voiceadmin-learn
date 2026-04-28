"""Agent entrypoint: FastAPI app hosting Twilio Media Streams over a WebSocket.

Run: `uv run uvicorn agent.main:app --host 0.0.0.0 --port 8000`

Flow per inbound call:
  1. Twilio hits POST /twiml → we reply with <Connect><Stream url=wss://.../ws/>
  2. Twilio opens WSS to /ws → we read the first "start" message to grab
     `streamSid` + `callSid`, then build the transport + pipeline.
  3. Pipeline = transport.input() → Deepgram STT → StateMachineProcessor → Cartesia TTS → transport.output()
  4. StateMachineProcessor spawns/stops the CallSessionRunner on StartFrame/EndFrame.

Pair this app with `ngrok http 8000` to expose a public URL to Twilio.
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from twilio.rest import Client as TwilioRestClient

from agent import tools
from agent.call_session import CallSessionRunner, IVRLLMClient
from agent.llm_client import AnthropicRepClient
from agent.logging_config import configure_logging, log
from agent.processors.state_processor import StateMachineProcessor
from agent.schemas import CallSession, IVRTurnResponse, PatientInfo, Turn

load_dotenv()
configure_logging()

app = FastAPI()

# Twilio μ-law at 8kHz. Override via env if you rewire the stream.
TWILIO_SAMPLE_RATE = int(os.getenv("TWILIO_SAMPLE_RATE", "8000"))

_IVR_SYSTEM_PROMPT = (
    "You are an IVR navigation agent. Use the provided tools to send DTMF, "
    "speak briefly when needed, record benefits the IVR reads aloud, and "
    "transfer to the rep when the IVR signals a handoff."
)
_REP_PROMPT_TEMPLATE = (Path(__file__).parent / "prompts" / "rep_turn.v1.txt").read_text()


def _default_patient() -> PatientInfo:
    return PatientInfo(
        member_id=os.getenv("PATIENT_MEMBER_ID", "M123456"),
        first_name=os.getenv("PATIENT_FIRST_NAME", "Alice"),
        last_name=os.getenv("PATIENT_LAST_NAME", "Example"),
        dob=os.getenv("PATIENT_DOB", "1980-05-12"),
    )


def _rep_system_prompt(patient: PatientInfo) -> str:
    return _REP_PROMPT_TEMPLATE.format(
        patient_name=f"{patient.first_name} {patient.last_name}",
        member_id=patient.member_id,
        patient_dob=patient.dob,
    )


@functools.cache
def _twilio_rest_client() -> TwilioRestClient:
    return TwilioRestClient(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])


class _UnimplementedIVRClient:
    """Fail-fast placeholder for the Groq tool-calling IVR client.

    Raises at construction so a misconfigured live deployment dies before
    audio starts flowing, not several seconds into the call. Replace with
    a real `IVRLLMClient` implementation before wiring live calls.
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "IVR tool-calling client is not wired. Replace `_UnimplementedIVRClient()` "
            "in agent/main.py with a real IVRLLMClient before accepting live calls."
        )

    async def complete_with_tools(
        self,
        system: str,
        history: list[Turn],
        tools: list[dict[str, object]],
        temperature: float = 0.1,
    ) -> IVRTurnResponse:
        raise NotImplementedError


@app.post("/twiml")
async def twiml(request: Request) -> Response:
    """Return TwiML that tells Twilio to open a Media Stream back to our /ws."""
    host = request.headers.get("host", "localhost")
    # Twilio Media Streams requires wss://; hardcode it. request.url.scheme
    # reports "http" behind ngrok/any reverse proxy even when the client used
    # https, and Twilio will reject a ws:// stream URL outright.
    stream_url = f"wss://{host}/ws"
    body = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<Response><Connect><Stream url="{stream_url}"/></Connect></Response>'
    )
    log.info("twiml_served", stream_url=stream_url)
    return Response(content=body, media_type="application/xml")


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    """Accept Twilio Media Streams, run the voice agent pipeline until hang-up."""
    await websocket.accept()
    while True:
        raw = await websocket.receive_text()
        msg = json.loads(raw)
        if msg.get("event") == "start":
            start = msg["start"]
            stream_sid: str = start["streamSid"]
            call_sid: str = start.get("callSid") or stream_sid
            log.info("ws_stream_started", call_sid=call_sid, stream_sid=stream_sid)
            break

    serializer = TwilioFrameSerializer(
        stream_sid=stream_sid,
        call_sid=call_sid,
        account_sid=os.environ.get("TWILIO_ACCOUNT_SID"),
        auth_token=os.environ.get("TWILIO_AUTH_TOKEN"),
    )
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=TWILIO_SAMPLE_RATE,
            audio_out_sample_rate=TWILIO_SAMPLE_RATE,
            add_wav_header=False,
            serializer=serializer,
        ),
    )

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        voice_id=os.environ.get("CARTESIA_VOICE_ID", "79a125e8-cd45-4c13-8a67-188112f4dd22"),
    )

    patient = _default_patient()
    session = CallSession(call_sid=call_sid, patient=patient)
    ivr_llm: IVRLLMClient = _UnimplementedIVRClient()
    rep_llm = AnthropicRepClient()
    runner = CallSessionRunner(
        session=session,
        ivr_llm=ivr_llm,
        rep_llm=rep_llm,
        tool_dispatcher=tools.dispatch,
        ivr_system_prompt=_IVR_SYSTEM_PROMPT,
        rep_system_prompt=_rep_system_prompt(patient),
        tools=[],
        twilio_client=_twilio_rest_client(),
    )
    state_proc = StateMachineProcessor(runner)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            state_proc,
            tts,
            transport.output(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=TWILIO_SAMPLE_RATE,
            audio_out_sample_rate=TWILIO_SAMPLE_RATE,
        ),
    )

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnect(_transport, _client):
        log.info("ws_client_disconnected", call_sid=call_sid)
        await task.cancel()

    pipeline_runner = PipelineRunner()
    await pipeline_runner.run(task)
    log.info("ws_pipeline_finished", call_sid=call_sid)
