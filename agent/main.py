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
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from twilio.rest import (  # pyright: ignore[reportMissingTypeStubs] (twilio SDK ships no type stubs)
    Client as TwilioRestClient,
)

from agent import tools
from agent.call_session import CallSessionRunner, IVRLLMClient
from agent.llm_client import AnthropicRepClient, GroqToolCallingClient
from agent.logging_config import configure_logging, log
from agent.processors.state_processor import StateMachineProcessor
from agent.schemas import CallSession, PatientInfo

load_dotenv()
configure_logging()

app = FastAPI()

# Twilio μ-law at 8kHz. Override via env if you rewire the stream.
TWILIO_SAMPLE_RATE = int(os.getenv("TWILIO_SAMPLE_RATE", "8000"))

_IVR_SYSTEM_PROMPT = (
    "You are a provider's-office agent calling a payer's IVR to verify "
    "patient eligibility. The user message you receive is one complete IVR "
    "utterance — typically a menu listing several numbered options, or a "
    "prompt asking for information. Treat it as input, NOT something to "
    "echo back. Every action is exactly one tool call; never reply in "
    "natural language.\n"
    "\n"
    "Goal: navigate the IVR to verify member benefits, providing requested "
    "patient details, and accept a transfer to a representative.\n"
    "\n"
    "How to act on each turn:\n"
    "- One IVR utterance = one menu (or one prompt) = one tool call from you.\n"
    "- If the utterance lists numbered options ('press 1 for X, press 2 for "
    "Y, press 3 for Z'), pick the SINGLE digit that best advances the goal "
    "and call `send_dtmf` with just that one digit. The IVR will play the "
    "next menu after you press; that next menu will arrive as a new turn.\n"
    "- If the utterance asks for information (member ID, DOB, patient name), "
    "call `speak` with ONLY that data — no preamble, no acknowledgement.\n"
    "- If the utterance is reading benefit details, call `record_benefit` to "
    "capture them.\n"
    "- If the utterance signals a handoff ('one moment', 'connecting you', "
    "'please hold'), call `transfer_to_rep`.\n"
    "- If the call is over ('thank you, goodbye'), call `complete_call`.\n"
    "- Only call `fail_with_reason` when the IVR is truly stuck (silent or "
    "looping with no usable options).\n"
    "\n"
    "Tools:\n"
    "- `send_dtmf(digits)`\n"
    "- `speak(text)`\n"
    "- `record_benefit(...)`\n"
    "- `transfer_to_rep()`\n"
    "- `complete_call(reason)`\n"
    "- `fail_with_reason(reason)`\n"
    "Output exactly one tool call per turn."
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
    tts = ElevenLabsTTSService(
        api_key=os.environ["ELEVENLABS_API_KEY"],
        # Default voice = "Sarah" (mature, reassuring premade voice — fits a
        # provider's-office persona). Override via env for a different voice.
        voice_id=os.environ.get("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL"),
    )

    patient = _default_patient()
    session = CallSession(call_sid=call_sid, patient=patient)
    ivr_llm: IVRLLMClient = GroqToolCallingClient()
    rep_llm = AnthropicRepClient()
    runner = CallSessionRunner(
        session=session,
        ivr_llm=ivr_llm,
        rep_llm=rep_llm,
        tool_dispatcher=tools.dispatch,
        ivr_system_prompt=_IVR_SYSTEM_PROMPT,
        rep_system_prompt=_rep_system_prompt(patient),
        tools=tools.groq_tool_schemas(),
        twilio_client=_twilio_rest_client(),
    )
    state_proc = StateMachineProcessor(runner)

    # VAD sits between transport.input() and STT. It emits
    # `VADUserStartedSpeakingFrame` when speech begins, which the state
    # processor uses as the barge-in signal (mark_interrupted → cancel
    # in-flight turn + drain queues). Without it, rep-mode interruption
    # never fires.
    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer())

    pipeline = Pipeline(
        [
            transport.input(),
            vad,
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

    # `event_handler` is dynamic on Pipecat's transport; no type stub exists.
    @transport.event_handler("on_client_disconnected")  # pyright: ignore[reportUnknownMemberType]
    async def on_disconnect(_transport: object, _client: object) -> None:
        log.info("ws_client_disconnected", call_sid=call_sid)
        await task.cancel()

    # Referenced here so pyright's `reportUnusedFunction` (strict mode)
    # doesn't fire — the decorator returns None, so without this line the
    # inner `async def` looks orphaned to the type checker.
    _ = on_disconnect

    pipeline_runner = PipelineRunner()
    await pipeline_runner.run(task)
    log.info("ws_pipeline_finished", call_sid=call_sid)
