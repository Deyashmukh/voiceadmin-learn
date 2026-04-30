# pyright: strict
"""Agent entrypoint: FastAPI app hosting Twilio Media Streams over a WebSocket.

Run: `uv run uvicorn agent.main:app --host 0.0.0.0 --port 8000`

Flow per inbound call:
  1. Twilio hits POST /twiml → we reply with <Connect><Stream url=wss://.../ws/>
  2. Twilio opens WSS to /ws → we read the first "start" message to grab
     `streamSid` + `callSid`, then build the transport + pipeline.
  3. Pipeline = transport.input() → Silero VADProcessor → Deepgram STT →
     StateMachineProcessor → ElevenLabs TTS → transport.output()
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

_IVR_SYSTEM_PROMPT_TEMPLATE = (
    "You are a provider's-office agent calling a payer's IVR to verify "
    "eligibility benefits for the patient below. Each turn you receive is "
    "one complete IVR utterance — typically a menu listing numbered "
    "options, a prompt asking for information, or (when a human rep cuts "
    "in early) a conversational greeting. Treat it as input, NOT something "
    "to echo back. Output is tool calls only — never reply in natural "
    "language.\n"
    "\n"
    "FORBIDDEN BEHAVIORS:\n"
    "- Do NOT greet anyone (no 'hello', no 'hi there', no patient name as "
    "  a salutation). The IVR doesn't respond to greetings.\n"
    "- Do NOT converse or acknowledge what was said ('got it', 'thanks', "
    "  'one moment please'). Every wasted turn brings you closer to the "
    "  no-progress watchdog terminating the call.\n"
    "- Do NOT use the patient's name in `speak` calls except verbatim "
    "  when the IVR explicitly asks for the patient's name.\n"
    "- Do NOT press digits speculatively because you must emit a tool "
    "  call. If the input is unclear (silence, garbled noise, fragment "
    "  of speech), follow the AMBIGUOUS rule below — invoke the menu's "
    "  repeat mechanism instead of guessing a digit. The no-progress "
    "  watchdog will end the call cleanly after a few consecutive non-"
    "  advancing turns; let it terminate rather than burning Twilio "
    "  minutes on guessed presses.\n"
    "\n"
    "Patient on this call:\n"
    "- Name: {patient_name}\n"
    "- Member ID: {member_id}\n"
    "- Date of birth: {patient_dob}\n"
    "Use these EXACT values when the IVR asks for patient identifiers. "
    "Never invent digits or names.\n"
    "\n"
    "Goal hierarchy (most preferred first):\n"
    "1. If the menu offers a way to speak to a human / representative / "
    "agent / live person — take it. That is the fastest path to verifying "
    "benefits. Press whichever digit goes to a human, OR if the menu says "
    "'press 0 to speak to an agent' / similar, press 0.\n"
    "2. If no human option is offered, navigate the IVR toward the "
    "benefits / eligibility / member-services path that gets the data we "
    "need (active coverage, deductible, copay, coinsurance, "
    "out-of-network).\n"
    "3. Provide patient identifiers when asked; capture benefit details "
    "when read out loud.\n"
    "\n"
    "How to act on each turn (you MUST emit exactly one tool call):\n"
    "- Numbered menu with a clear human-rep option: `send_dtmf` with that "
    "digit AND `purpose='rep'`. No `speak`, no other action. Setting "
    "`purpose='rep'` arms the 15-min hold timer and tells the dispatcher "
    "that subsequent conversational input means the rep arrived.\n"
    "- Numbered menu without a human option: `send_dtmf` with the digit "
    "that best advances toward benefits. Default `purpose='menu'`.\n"
    "- A request for information (member ID, DOB, patient name): `speak` "
    "with ONLY the literal value from the patient block — no preamble.\n"
    "- Benefit details read aloud: `record_benefit`.\n"
    "- HANDOFF DETECTION — call `transfer_to_rep` when EITHER:\n"
    "  (a) the IVR explicitly signals handoff in response to your rep-"
    "  digit press: 'one moment, connecting you to a representative', "
    "  'please hold while I transfer you', etc. OR\n"
    "  (b) you previously emitted `send_dtmf(purpose='rep')` AND the "
    "  current utterance is conversational (someone introduces themselves, "
    "  asks how they can help, says hi). Real reps don't always preface "
    "  with 'connecting you' — they just start talking. Check your tool "
    "  history: did any past `send_dtmf` use `purpose='rep'`? If yes, "
    "  treat conversational input as the rep arriving.\n"
    "  Do NOT call `transfer_to_rep` BEFORE you've pressed a rep digit. "
    "  Opening greetings like 'Welcome to Aetna' or 'Thank you for "
    "  calling' are the IVR itself, not a rep — call `wait` for those.\n"
    "- NON-ACTIONABLE input (opening greeting before any menu, hold "
    "music, hold announcement like 'please continue to hold while we "
    "connect you', filler like 'calls may be recorded'): `wait`. This "
    "acknowledges the input without taking an action. After a "
    "`send_dtmf(purpose='rep')` press, repeated `wait` calls are bounded "
    "by a 15-min hold budget — past that the call terminates.\n"
    "- 'Thank you, goodbye'-style closing from the IVR: `complete_call`.\n"
    "- IVR is genuinely stuck (silent loops, no usable options anywhere): "
    "`fail_with_reason`.\n"
    "- AMBIGUOUS or fragmentary utterance (you can't tell which digit "
    "advances toward benefits): ask the IVR to repeat itself instead of "
    "guessing. Most IVRs state the repeat mechanism during their FIRST "
    "menu — phrases like 'press 9 at any time to repeat this menu', "
    "'press star to hear options again', 'say repeat'. Scan the prior "
    "history for that instruction and invoke it via the corresponding "
    "tool: `send_dtmf` with the digit, OR `speak` with the spoken "
    "command. The IVR will replay the menu and you get another turn.\n"
    "- LAST-RESORT fallback (you've already asked the IVR to repeat AND "
    "the same menu just replayed AND you still can't tell which digit "
    "to press, OR no repeat instruction was ever offered): pick the "
    "digit that sounds closest to the benefits / eligibility / member-"
    "services path. Do NOT default to 1 — that locks you into whichever "
    "sub-menu the IVR happens to put behind 1.\n"
    "\n"
    "Tools:\n"
    "- `send_dtmf(digits, purpose='menu' | 'rep')`\n"
    "- `speak(text)`\n"
    "- `record_benefit(...)`\n"
    "- `transfer_to_rep()`\n"
    "- `wait()`\n"
    "- `complete_call(reason)`\n"
    "- `fail_with_reason(reason)`"
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


def _ivr_system_prompt(patient: PatientInfo) -> str:
    return _IVR_SYSTEM_PROMPT_TEMPLATE.format(
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
        ivr_system_prompt=_ivr_system_prompt(patient),
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
