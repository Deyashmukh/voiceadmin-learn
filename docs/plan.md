# Hybrid Voice Agent Harness — Eligibility Verification Learning Project (v2)

## Context

After a deep-dive on Revenue Cycle Management (RCM) and how production voice agents like VoiceAdmin work, the goal is to build a **working voice agent end-to-end** — not for production, but to internalize how these systems are actually architected. Specifically: how an LLM-with-tools layer drives deterministic side effects (DTMF, TTS, structured extraction), how **VAD + barge-in** creates the illusion of natural conversation, and how this stack touches real telephony.

The target workflow is **Eligibility & Benefits Verification** — the canonical RCM voice task. The agent calls a "payer," authenticates with provider info, navigates an IVR menu, transitions to a human rep when handed off, holds a natural conversation to extract benefits, and ends the call cleanly with structured JSON: `{active, deductible_remaining, copay, coinsurance, out_of_network_coverage}`.

To keep the project focused on architecture (not HIPAA/legal plumbing), the agent dials the user's SMS-verified personal cell over real Twilio telephony. The user roleplays both the IVR menu (reading prompts aloud while the agent's tool calls send DTMF the user can hear on their cell) and the human rep (scripted benefits read with optional small-talk deviation). This gives the full real-world loop (audio streams, DTMF tones, VAD interrupting TTS, latency pressure) without the risks of calling a real payer or the overhead of building a second Twilio service.

Success = the agent autonomously navigates a DTMF menu via tool calls, detects the handoff to a human, holds a natural conversation, extracts structured benefits, and ends the call gracefully. Bonus: the human rep occasionally deviates from the script, exercising the conversational LLM's recovery path.

## Architecture pivot — why this plan is v2

The pre-pivot plan used a LangGraph state machine with a regex-based IVR classifier (shipped as M5/T3 + M5/T4, see PR #4). After feedback from someone solving this in production, two things changed:

- **Per-payer regex doesn't scale.** Real-world deployments touch dozens of payer IVRs, each with idiosyncratic phrasing that drifts over time. Maintaining one classifier per payer is operational debt. A single LLM-with-tools dispatcher handles N payers with one prompt.
- **LangGraph wasn't earning its keep.** The pre-pivot graph had 6 nodes, 1 conditional edge, single-step turns, no checkpointer. That's a `match` statement with extra abstraction. The two-mode architecture below has a turn loop and no graph at all.

Pivot summary:

- **IVR side**: LLM-with-tools loop. Tools constrain the action space; argument validators at the dispatch boundary engineer the determinism the LLM doesn't provide on its own.
- **Rep side**: structured-output LLM (`RepTurnOutput`). Conversation, not call-and-dispatch.
- **Mode flip**: one-way `ivr → rep` triggered by the IVR LLM's `transfer_to_rep()` tool call. Hold music drops out via Deepgram's VAD; no separate `wait` mode.
- **No state graph.** A `CallSession` dataclass + a per-turn loop in `agent/call_session.py` replaces `agent/graph.py` and `agent/graph_runner.py`.
- **No regex classifier.** `agent/classifier.py` retires.
- **No second Twilio number.** User dials own cell; user is the IVR + rep.

The mock payer (`mock_payer/*` from M5/T3) and the rule-based classifier (`agent/classifier.py` from M5/T4) were deleted from the working tree post-pivot but remain in git history as **learning artifacts** — they document what the regex-based approach looked like, and the contrast against the new architecture is itself the lesson.

## Stack (April 2026)

| Layer | Choice | Why |
|---|---|---|
| **Voice pipeline framework** | **Pipecat** (Python) | Handles Twilio WebSocket transport, VAD, barge-in, TTS interruption, turn-taking. The pipeline composition with `CallSession` (alongside, not embedded) is the architectural non-negotiable. |
| **Call orchestration** | **`agent/call_session.py`** — plain Python turn loop driven by an `asyncio.Task` per call | Replaces LangGraph. The new shape has one node (the LLM call) and a self-loop; that's not a graph. Plain code is more debuggable for a learning project, and the pedagogical lesson "what would a state-graph framework have bought me?" is exactly the lesson worth ending the project with. |
| **IVR LLM (tool-calling)** | **Llama 4 Scout on Groq** (`meta-llama/llama-4-scout-17b-16e-instruct`) | 17B active params (MoE, 16 experts, 400B total). Native strict JSON schema + tool use. Sub-100ms TTFT on Groq's LPU. Cheap (~$0.11/$0.34 per M tokens). Already pinned. |
| **Rep LLM (conversational, structured output)** | **Claude Haiku 4.5 via Anthropic API** (`claude-haiku-4-5-20251001`) | Voice-agent benchmarks specifically call it out for following long system prompts literally and staying calm on long conversations. Native structured output via Anthropic's `tools` mechanism. ~400-800ms typical latency on Anthropic's API. Different vendor than Groq — exposes the project to Anthropic SDK patterns. |
| **Telephony** | **Twilio** (Media Streams for audio; REST API `sendDigits` for mid-call DTMF) | Trial tier sufficient — agent dials user's SMS-verified personal cell. No second number. |
| **ASR** | **Deepgram Nova-3** (streaming WebSocket) | Sub-300ms, strong VAD/endpointing. `endpointing` + `utterance_end_ms` tuned in M2. Hold-music dropped at the VAD layer — no transcripts during silence between IVR transition and rep pickup. |
| **TTS** | **Cartesia Sonic-3** | 40ms time-to-first-audio. Critical for barge-in feel. |
| **Observability** | **Langfuse** (self-hosted, Docker Compose) via `@observe` decorators | The LangChain callback handler path retires alongside LangGraph. New: explicit `@observe` on the per-turn function, on each LLM call, on each tool dispatcher. `langfuse_session_id = call_sid` set on the root span groups all turns of one call into one trace tree. |
| **Runtime** | Python 3.12 + `asyncio` + `uv` for dep mgmt | |

**Realistic cost estimate: $25–$40** (Twilio minutes during debugging + TTS during barge-in + Groq + Anthropic Haiku spend).

## Software hygiene baseline (unchanged from M1)

- **Tooling:** `ruff` (lint + format), `pyright` basic mode, `pytest` + `pytest-asyncio`, `pre-commit` running ruff + pyright before every commit.
- **Dependency pinning:** `pyproject.toml` pins exact versions. `uv.lock` committed.
- **Structured logging:** `structlog` with `call_sid` and `turn_index` bound as contextvars. JSONRenderer in production; `configure_logging()` invoked at agent startup.
- **Secrets:** `.env` is gitignored. `.env.example` checked in with placeholders.
- **Outbound dial allowlist:** `ALLOWED_DESTINATIONS` env var + a hard check in `dial()` before every `calls.create()`. Non-negotiable.
- **Testing seams:** all external dependencies (LLM clients, Twilio client, ASR) sit behind typed `Protocol` interfaces so unit tests inject fakes. Unit tests run with **zero network calls**.

**Cut as YAGNI:** GitHub Actions CI, pyright strict, Docker for the agent itself, OpenTelemetry/Prometheus, `tenacity`, multiple env profiles, mock payer (retired post-pivot).

## Architectural non-negotiables (post-pivot)

- **`CallSession` runs alongside the Pipecat pipeline, never inside a `FrameProcessor`'s `process_frame()`.** Embedding the LLM call in the frame path blocks the audio loop and destroys barge-in. The Pipecat adapter queues transcripts and forwards interrupt events; the loop runs in a separate `asyncio.Task`.
- **Interrupts are real `asyncio.Task.cancel()` calls, not flag checks.** Setting `session.interrupted = True` does not cancel an in-flight LLM call. Handlers must be cancellation-safe.
- **Bounded queues.** `in_queue` uses drop-oldest on full (stale transcripts are worthless). `out_queue` blocks on full (TTS backpressure is desired).
- **One `CallSession` per call**, spawned on Pipecat transport-connect, stopped on transport-disconnect. Never process-global.
- **Determinism is engineered at the tool-dispatch boundary, not assumed from the LLM.** Every tool call has an arg validator. `send_dtmf("9")` when the most recent menu didn't offer 9 → reject, append a tool-error message to history, let the LLM re-pick. No exceptions.
- **Mode is one-way: `ivr → rep`.** If a rep puts the agent on hold and an IVR comes back, the agent stays in rep mode. Acceptable for learning; flagged for production.

## Two-mode CallSession architecture

### Per-turn loop (the entire orchestration)

```python
async def _consume(self) -> None:
    structlog.contextvars.bind_contextvars(call_sid=self.call_ctx.call_sid, turn_index=0)
    while not self.session.done:
        transcript = await self.in_queue.get()
        # Drain stale transcripts that piled up during the prior turn.
        while not self.in_queue.empty():
            transcript = self.in_queue.get_nowait()

        structlog.contextvars.bind_contextvars(turn_index=self.session.turn_count)
        self._current_turn = asyncio.create_task(self._run_turn(transcript))
        try:
            await self._current_turn
        except asyncio.CancelledError:
            if self._interrupt_requested:
                self._interrupt_requested = False
                continue
            raise

async def _run_turn(self, transcript: str) -> None:
    self.session.history.append(Turn(role="user", content=transcript))
    if self.session.mode == "ivr":
        await self._ivr_turn()
    else:
        await self._rep_turn()
    self.session.turn_count += 1
```

### IVR mode

```python
async def _ivr_turn(self) -> None:
    response = await self.ivr_llm.complete_with_tools(
        system=IVR_SYSTEM_PROMPT,
        history=self.session.history,
        tools=IVR_TOOLS,
        temperature=0.1,
    )
    advanced = False
    for call in response.tool_calls:
        result = await self.tool_dispatcher.dispatch(call, self.session)
        self.session.history.append(Turn(role="tool", call=call, result=result))
        if result.side_effect:
            await self.actuator.execute(result.side_effect)
        if result.advanced_call_state:
            advanced = True
    if not advanced:
        self.session.ivr_no_progress_turns += 1
        if self.session.ivr_no_progress_turns >= 2:
            self.session.done = True
            self.session.completion_reason = "ivr_no_progress"
    else:
        self.session.ivr_no_progress_turns = 0
```

**IVR tools (Pydantic argument schemas):**

```python
class SendDTMFArgs(BaseModel):
    digits: str  # validated: digits in last menu options OR universal (#, *)

class SpeakArgs(BaseModel):
    text: str = Field(min_length=1, max_length=200)

class RecordBenefitArgs(BaseModel):
    field: Literal["active", "deductible_remaining", "copay", "coinsurance", "out_of_network_coverage"]
    value: bool | float | None  # validated against field type

class TransferToRepArgs(BaseModel):
    pass  # no args; flips session.mode to "rep"

class CompleteCallArgs(BaseModel):
    reason: Literal["benefits_extracted", "ivr_dead_end", "user_hangup"]

class FailWithReasonArgs(BaseModel):
    reason: str = Field(min_length=1, max_length=120)
```

**Validation responsibilities live in `agent/tools.py`:**

- `send_dtmf`: digit must be in `session.recent_menu_options` if last transcript was a menu prompt; universal keys (`#`, `*`) always allowed.
- `record_benefit`: value type matches field type; numeric fields must be non-negative; `active` and `out_of_network_coverage` must be bool.
- `transfer_to_rep`: no-op validation; flips `session.mode = "rep"`.
- `complete_call` / `fail_with_reason`: set `session.done = True` and capture the reason.
- Failed validation returns a structured tool-error message that goes back into history so the LLM can re-pick on the next iteration.

### Rep mode

```python
class RepTurnOutput(BaseModel):
    reply: str  # what the agent says aloud ("" = stay silent)
    extracted: Benefits  # partial; only non-None fields are merged
    phase: Literal["extracting", "complete", "stuck"]
    reasoning: str | None = None

async def _rep_turn(self) -> None:
    output = await self.rep_llm.complete_structured(
        system=REP_PERSONA_PROMPT,
        history=self.session.history,
        schema=RepTurnOutput,
    )
    # Non-None merge into session.benefits.
    for field, value in output.extracted.model_dump(exclude_none=True).items():
        setattr(self.session.benefits, field, value)
    if output.reply:
        await self.actuator.execute(SpeakIntent(text=output.reply))
    self.session.history.append(
        Turn(role="assistant", content=output.reply, extracted=output.extracted)
    )
    if output.phase == "complete":
        self.session.done = True
        self.session.completion_reason = "rep_complete"
    elif output.phase == "stuck":
        self.session.stuck_turns += 1
        if self.session.stuck_turns >= 2:
            self.session.done = True
            self.session.completion_reason = "rep_stuck"
    else:
        self.session.stuck_turns = 0
```

**Rep persona prompt** (lives in `agent/prompts/rep_turn.v1.txt`):

> You are Morgan, a staff member at a provider's office calling an insurance rep to verify eligibility benefits for {patient}. You are a real person on a real phone call — not a robot, no script. If the rep makes small talk, engage briefly. If they say something difficult, acknowledge with warmth before continuing. Never announce your own intent (*"I need to..."*, *"Let me get back to..."*).
>
> When all required Benefits fields are confidently filled, set `phase: "complete"` and produce a brief, friendly closing reply — *"That's everything I needed, thanks Sam, have a great rest of your day."* — then stop asking questions. If the rep volunteers extra info after your close, ack briefly and stay `phase: "complete"`.
>
> If you've asked for the same field twice without a usable answer, set `phase: "stuck"` and produce a brief apology + closing line.

### Mode flip mechanics

The flip happens during the IVR LLM's tool dispatch:

```python
# in agent/tools.py
async def dispatch(self, call: ToolCall, session: CallSession) -> ToolResult:
    if call.name == "transfer_to_rep":
        session.mode = "rep"
        return ToolResult(
            success=True,
            advanced_call_state=True,
            message="Mode flipped to rep. Next turn routes to rep_turn LLM.",
        )
    # ... other tools
```

The current IVR turn finishes. The next iteration of `_consume` reads the next transcript and routes via `if session.mode == "ivr"` — now false, so `_rep_turn()` runs.

### Hold music handled implicitly

Deepgram's VAD doesn't emit transcripts during music or silence. The `_consume` loop sits at `await self.in_queue.get()` with no work to do. Free.

If a hold *announcement* gets transcribed (e.g., *"please continue to hold"*), it arrives in rep mode and the rep LLM handles it — most likely `phase: "extracting"` with `reply: ""` (stay silent).

### Interrupts

`StateMachineProcessor.process_frame` receives a `VADActiveFrame` while TTS is playing → calls `session.mark_interrupted()` → cancels `_current_turn` task → the LLM `await` raises `CancelledError` inside `_run_turn` → `_consume` catches it, drains stale transcripts, picks up next turn cleanly. Same as the pre-pivot design.

## Failure modes

| Integration | Failure | Detection | Behavior |
|---|---|---|---|
| Twilio Media Streams | WebSocket disconnect mid-call | Pipecat transport close event | Log `call_sid` + last state, mark call `terminated_abnormal`, no retry |
| Twilio REST `sendDigits` | 4xx / network error | Exception in actuator | Retry once with 500ms backoff; on second failure → `fail_with_reason("dtmf_dispatch_failed")` |
| Deepgram ASR | WebSocket drop mid-utterance | Pipecat error frame | Reconnect once; on second failure → `fail_with_reason("asr_lost")` |
| Groq (Llama 4 Scout, IVR LLM) | 429 / 5xx / >4s timeout | HTTP error or timeout | One retry on 429; on timeout / 5xx → IVR turn produces no tool calls; CallSession watchdog (no progress 2 turns) → fail |
| Anthropic (Haiku 4.5, rep LLM) | 429 / 5xx / >4s timeout | HTTP error or timeout | One retry on 429; on timeout / 5xx → rep turn produces empty reply; watchdog catches |
| Cartesia TTS | First-byte timeout | 2s timeout | No retry; log and skip the spoken reply |
| Tool argument validation | Out-of-range digit, wrong type, etc. | Validator returns error | Append tool-error to history; LLM re-picks on next turn |
| LLM hallucinated tool call | Tool name not in registry | Dispatcher catches | Append "unknown tool" error; LLM re-picks |
| Tool dispatch raised | Unexpected exception in dispatcher | Try/except in `dispatch()` | Log with tool name; route to `fail_with_reason("tool_dispatch_exception")` |
| Watchdog (IVR no progress) | 2 IVR turns with no advancing tool call | Counter in CallSession | `fail_with_reason("ivr_no_progress")` |
| Rep stuck | `phase: "stuck"` for 2 consecutive turns | Counter | End call gracefully with apology line |

**Per-call budgets:** Groq timeout = 4s. Anthropic timeout = 4s. Cartesia first-byte = 2s. Deepgram utterance-end = configured explicitly.

## Build order

### Done (M1–M4)

1. **M1 ✅** — Env, keys, hygiene baseline (uv pinned, ruff, pyright basic, pytest, pre-commit, structlog, dialer allowlist).
2. **M2 ✅** — Pipecat hello-world + structured logging + barge-in (cancel-to-silence < 150ms verified).
3. **M3 ✅** — LangGraph state machine + GraphRunner (offline, 41 unit tests, zero network). *Retiring in M5'/E.*
4. **M4 ✅** — Wire GraphRunner into Pipecat + Langfuse callback handler. *LangChain callback handler retiring in M5'/F.*

### Done as learning artifact (PR #4)

5. **M5/T3 ✅** — Mock payer (FastAPI + TwiML Gather tree). Stays in history; not wired into anything post-pivot.
6. **M5/T4 ✅** — `ivr_nav` regex classifier (4 prompt shapes). Stays in history as the contrast against M5'.

### New milestones

7. **M5'/A — Schemas.** New types in `agent/schemas.py`: `CallSession`, `Turn`, `RepTurnOutput`, tool-arg models. Retire `ClassifierResult`, `IVRClassifier` Protocol. **Relax `Benefits.active: bool` to `bool | None = None`** so partial extraction in rep mode (where the field hasn't been heard yet) is representable — currently `active` is required, which would force every interim merge to fail validation.

8. **M5'/B — Tools layer + DTMF actuator helper.** `agent/tools.py`: registry of IVR tools as Pydantic models, dispatcher with per-tool validators. Pure functions; tool dispatch returns `ToolResult` plus an optional `SideEffectIntent` (DTMF / Speak / Hangup) that the actuator executes. Also build `agent/telephony/dtmf.py`: thin wrapper around `twilio_client.calls(sid).update(twiml="<Response><Play digits=\"...\"/></Response>")` for mid-Media-Stream DTMF injection. Twilio's REST `<Play digits>` mid-call is the documented path; the dialer's existing allowlist doesn't apply (call already exists). Unit-test the dispatcher's validators in M5'/G; the dtmf helper proves itself in M6'.

9. **M5'/C — Anthropic SDK + rep LLM client.** Pin `anthropic` in `pyproject.toml`. Extend `agent/llm_client.py` with a Claude Haiku 4.5 backend that exposes `complete_structured(system, history, schema=RepTurnOutput)`. Implementation: Anthropic's SDK exposes `messages.parse(output_format=schema)` which validates the response against the Pydantic model and returns a `parsed_output` instance directly — no manual `messages.create(tools=...)` + `model_validate` round-trip needed. Mark the persona system prompt with `cache_control: ephemeral` for forward compatibility. **Note on caching:** Haiku 4.5's minimum cacheable prefix is 4096 tokens. A typical persona prompt (a few hundred tokens) won't actually cache — the marker is harmless on short prompts (no error, just `cache_creation_input_tokens: 0`) and ready for a longer persona later. Verification waits for the cache assertion (see Verification section).

10. **M5'/D — CallSession (split into 3 commits).**
    - **D1**: scaffolding + IVR-only loop (mode hardcoded "ivr", watchdog). **Pipecat wiring**: actuator pushes spoken text into `session.out_queue` (the existing `state_processor` pump → `TextFrame` → Cartesia path is unchanged). Avoids coupling actuator to the FrameProcessor. **Carry forward the M4 `mark_interrupted` behavior**: drain `out_queue` *before* cancelling `_current_turn` so a turn that finished moments before barge-in doesn't still get spoken.
    - **D2**: rep-mode handler with structured output (mode hardcoded "rep").
    - **D3**: mode-aware routing + `transfer_to_rep` wiring (the flip).

11. **M5'/E — Retire LangGraph + classifier.** Remove `langgraph`, `langchain` from deps. Delete `agent/graph.py`, `agent/graph_runner.py`, `agent/classifier.py`. Delete corresponding tests. **Order matters**: this milestone runs *after* M5'/D3 lands AND after `agent/main.py` has been rewritten to construct a `CallSession` instead of a `GraphRunner` — `main.py` currently imports all three retiring modules, so deleting them earlier breaks the entrypoint. Update `agent/processors/state_processor.py` to wire `CallSession`'s queue interface (identical to `GraphRunner` — `submit_transcript`, `start`/`stop`, `mark_interrupted`, `out_queue` — so the change is largely a symbol swap).

12. **M5'/F — Observability rebuild.** `agent/observability.py` rewritten to use Langfuse `@observe` decorators on the per-turn function, both LLM call sites (IVR + rep), each tool dispatcher. Set `langfuse_session_id` via `update_current_trace`. Verification asserts the trace tree shows correct **span nesting** (per-turn span contains LLM-call spans contains tool-dispatch spans), not just span existence — `@observe` context across `asyncio.to_thread` worked in M4 via `get_client()` re-fetch; the same trick is needed for the Anthropic SDK call wrapper.

13. **M5'/G — Tests.** `tests/unit/test_call_session.py`, `tests/unit/test_tools.py`. New fakes for Anthropic + Groq tool clients. Zero network. Cover specifically: `ivr_no_progress` watchdog increments on a turn where the LLM produced *zero* tool calls (Groq timeout case) — easy to miss, easy to over-count.

14. **M6' — DTMF spike.** ~10 minutes once M5'/D lands. Place a call to your cell, drive the IVR LLM through one menu via your voice ("press 1 for eligibility"), confirm sendDigits tones audible on your cell. Document in `NOTES.md`.

15. **M7' — Manual phone test (5/5 happy path).** Place 5 live calls. User roleplays IVR (1-2 menus, data-entry prompt, "connecting you to a representative") then rep (greeting, scripted benefits read, closing). Each run: agent navigates IVR via tools, transitions cleanly, holds natural conversation, populates Benefits, ends call gracefully. Capture Langfuse trace IDs. 5/5 success rate is the M7' exit gate.

16. **M7'+ — Rep deviations (optional, only if M7' clean).** User adds three rep deviations across runs: small talk (weather), ambiguous answer (rep hedges), emotional content (rep mentions difficult day). Verify rep LLM handles each via persona prompting alone — no template logic, no banned-phrase lint.

## Key files (post-pivot)

```
voiceadmin-learn/
├── README.md
├── Makefile
├── pyproject.toml                    # langgraph + langchain OUT, anthropic IN
├── uv.lock
├── .env.example                      # adds ANTHROPIC_API_KEY
├── .pre-commit-config.yaml
├── docs/
│   └── plan.md                       # this file
├── agent/
│   ├── main.py                       # Pipecat pipeline + entrypoint (CallSession wired in)
│   ├── call_session.py               # NEW: per-turn loop, mode-aware dispatch (replaces graph.py + graph_runner.py)
│   ├── tools.py                      # NEW: IVR tool definitions + dispatcher + validators
│   ├── llm_client.py                 # Groq + Anthropic backends behind one Protocol
│   ├── observability.py              # @observe decorators; Langfuse SDK direct (no LangChain handler)
│   ├── processors/
│   │   └── state_processor.py        # Pipecat adapter; queues to/from CallSession
│   ├── telephony/
│   │   ├── dialer.py                 # unchanged
│   │   └── dtmf.py                   # NEW: Twilio REST mid-stream sendDigits wrapper (Calls(sid).update)
│   ├── actuator.py                   # NEW: executes IVR side-effect intents (DTMF, TTS, hangup)
│   ├── prompts/
│   │   ├── ivr_system.v1.txt         # NEW: IVR mode system prompt
│   │   └── rep_turn.v1.txt           # NEW: rep mode persona prompt
│   ├── logging_config.py             # unchanged
│   └── schemas.py                    # CallSession, Turn, RepTurnOutput, tool-arg models; ClassifierResult OUT
└── tests/
    └── unit/
        ├── conftest.py               # FakeGroqToolClient, FakeAnthropicClient (replaces FakeLLMClient/FakeClassifier)
        ├── test_call_session.py      # NEW: mode flip, validator retry, merge, termination
        ├── test_tools.py             # NEW: per-validator coverage
        ├── test_state_processor.py   # updated: wires CallSession
        ├── test_dialer.py            # unchanged
        └── test_schemas.py           # unchanged
```

## Verification

- **M5'/A:** `pytest tests/unit/test_schemas.py` passes; new types instantiate; tool-arg validators reject malformed input.
- **M5'/B:** `pytest tests/unit/test_tools.py` passes; every validator hit on happy + invalid path; dispatcher returns tool-error messages on rejection.
- **M5'/C:** Anthropic Haiku 4.5 client reaches a real Anthropic endpoint in a manual REPL check; `messages.parse(output_format=RepTurnOutput)` returns a validated instance. Cancellation propagation through the SDK await verified offline. Prompt-cache assertion (`cache_creation_input_tokens > 0` / `cache_read_input_tokens > 0`) deferred until the persona prompt grows past Haiku's 4096-token minimum cacheable prefix — current persona is shorter, so the marker is set but caching is a no-op (silently, no error). Re-enable the assertion in M7' if the persona crosses that boundary.
- **M5'/D:** `pytest tests/unit/test_call_session.py` passes; cancellation test (slow-LLM barge-in) cancels within 150ms; `out_queue` is drained on `mark_interrupted` (regression for the M4 post-review fix); watchdogs trigger on no-progress and stuck cases.
- **M5'/E:** `pyproject.toml` no longer mentions `langgraph` or `langchain`. `agent/graph*.py` and `agent/classifier.py` deleted. `agent/main.py` constructs `CallSession`, not `GraphRunner`. Full unit suite green. (`mock_payer/` was retained at first then deleted in a follow-up cleanup PR — it survives in git history as a learning artifact.)
- **M5'/F:** Langfuse UI shows trace tree with one session per call; spans nest correctly (per-turn → LLM-call → tool-dispatch), not just spans existing in isolation.
- **M6':** DTMF tones audible on user's cell during a live Media Streams call. Logged in `NOTES.md`.
- **M7':** 5/5 happy-path runs land complete `Benefits`. Per-call total token budget under ~30k (sum across IVR + rep LLMs). Each run has a Langfuse trace.
- **M7'+:** Each deviation passes 5/5 runs. Logs show the rep LLM's reasoning fields explaining the chosen behavior.
- **Overall success metric:** 5 consecutive successful end-to-end runs with no manual intervention beyond playing the IVR + rep roles.

## Decisions made during planning (with pivot history)

- **Repo layout:** single repo, `agent/` only post-pivot (`mock_payer/` retired and deleted; lives in git history).
- **Hygiene baseline:** `uv` pinned, `ruff`, `pyright` basic, `pytest`, `pre-commit`. Established M1.
- **Observability:** Langfuse, self-hosted. Pre-pivot used LangChain callback handler; post-pivot uses `@observe` decorators directly. LangSmith was considered and rejected.
- **State machine framework:** *Pre-pivot:* LangGraph, runs alongside Pipecat. *Post-pivot:* dropped — the graph was barely using graph features and the new shape has no graph. Replaced with `CallSession` plain-Python loop. Sunk-cost cleared by recognizing the production-shape lesson is more valuable than the LangGraph-internals one.
- **IVR navigation:** *Pre-pivot:* rule-based regex classifier with planned LLM fallback at M7b. *Post-pivot:* LLM-with-tools from day one. Production feedback was that per-payer regex doesn't scale and an LLM-with-tools dispatcher generalizes.
- **Hybrid principle (revised):** the hybrid isn't *"LLM vs regex"* — it's *"tool-calling LLM with deterministic actuators"* on the IVR side and *"structured-output LLM with conversational reply"* on the rep side. Deterministic side effects are achieved via tool-arg validation, not by avoiding the LLM.
- **Persona for human phase:** the agent plays a provider-side staff member ("Morgan, staff at a provider's office") calling the payer to verify benefits — not a payer rep. Persona prompt forbids announcing intent, expects brief small-talk engagement, routes empathy through the same single LLM call.
- **Mode reversal:** `rep → ivr` is not supported. If a rep puts the agent on hold and an IVR comes back, the agent stays in rep mode. Acceptable for learning; flagged for production.
- **Hallucination guardrails:** prompt-only on the rep side. Tool-arg validation on the IVR side. No verifier pass over rep transcripts (M7'+ won't add it; production would).
- **Mock payer / second Twilio number:** retired. User dials own cell, roleplays both halves of the call.
- **Budget:** $25–$40 approved. Slight Anthropic Haiku cost added; offset by no second Twilio number.

## Execution rules (unchanged from CLAUDE.md)

1. Git worktrees per independently executable milestone or sub-task. Naming: `arch-pivot-plan`, `arch-pivot-tools`, `arch-pivot-call-session`, etc.
2. Parallelize aggressively when work has no shared state or sequential dependency.
3. Simplify before verifying — `superpowers:simplify` skill at end of every milestone.
4. Verify before claiming done — `superpowers:verification-before-completion`. Evidence = test/lint/typecheck output + quantitative checks.
5. Commit only after simplify + verify pass. One commit per logical sub-task.
