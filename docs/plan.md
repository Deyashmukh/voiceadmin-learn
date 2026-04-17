# Hybrid Voice Agent Harness — Eligibility Verification Learning Project

## Context

After a deep-dive conversation on Revenue Cycle Management (RCM) and how production voice agents like VoiceAdmin work, the goal is to build a **working hybrid voice agent end-to-end** — not for production, but to internalize how these systems are actually architected. Specifically: how a deterministic **state machine** (the "control plane") composes with a probabilistic **LLM** (the "reasoning engine"), how **VAD + barge-in** creates the illusion of natural conversation, and how this stack touches real telephony.

The target workflow is **Eligibility & Benefits Verification** — the canonical RCM voice task. The agent calls a "payer," authenticates with provider info, navigates an IVR, speaks to a mock human rep when handed off, and returns structured JSON: `{active, deductible_remaining, copay, coinsurance, out_of_network_coverage}`.

To keep the project focused on architecture (not HIPAA/legal plumbing), both sides of the call are under our control: **real Twilio telephony**, but the "insurance payer" on the other end is a second Twilio number running a mock IVR + scripted human rep we build ourselves. This gives the full real-world loop (audio streams, DTMF tones, VAD interrupting TTS mid-sentence, latency pressure) without the risks of calling a real payer.

Success = the agent autonomously dials the mock payer, correctly navigates a DTMF IVR, handles a human-rep handoff (including the human interrupting it), extracts structured benefits data, and outputs a clean JSON payload. Bonus: the human rep occasionally deviates from the script, forcing the hybrid fallback path to fire.

## Stack (researched April 2026)

| Layer | Choice | Why |
|---|---|---|
| **Voice pipeline framework** | **Pipecat** (Python) | Handles Twilio WebSocket transport, VAD, barge-in, TTS interruption, turn-taking out of the box. Lets us focus on hybrid control logic, not audio plumbing. |
| **State machine** | **LangGraph** (`StateGraph` with typed `TypedDict` state, explicit nodes, conditional edges) | Industry standard. You'll dig into the internals wherever something is opaque. **Critical:** LangGraph runs *alongside* the Pipecat pipeline, not embedded inside a FrameProcessor — see design section. |
| **Telephony** | **Twilio** (Media Streams for audio; REST API `sendDigits` or stream `dtmf` events for mid-call DTMF) | Industry standard. Two numbers: one for the agent, one for the mock payer. `<Play digits>` does **not** work inside an active Media Streams session — mid-stream DTMF injection must be verified explicitly (see M5). |
| **Twilio account tier** | **Pay-as-you-go for the agent account; trial is fine for the mock payer** | Trial restrictions only bite on outbound: prepended voicemail message, verified-destinations-only, DTMF timing drift. The mock payer only *receives* inbound calls, so trial is fine there. Budget ~$20 to upgrade the agent account only. |
| **ASR** | **Deepgram Nova-3** (streaming WebSocket) | Sub-300ms, strong VAD/endpointing. Tune `endpointing` and `utterance_end_ms` explicitly in M2 — don't leave defaults. |
| **TTS** | **Cartesia Sonic-3** | 40ms time-to-first-audio — current industry leader. Critical for barge-in feel. |
| **LLM (structured extraction)** | **Kimi K2 on Groq** (`moonshotai/kimi-k2-instruct`) | Groq's strict `response_format: json_schema` is only supported on Kimi K2, Llama 4 Maverick, and Llama 4 Scout. Qwen3-32B only supports loose `json_object`, which would force a validate-and-retry wrapper. Kimi K2 binds directly to Pydantic. ~$1/$3 per M tokens. Used ONLY for `extract_benefits`. |
| **LLM (clarification / reasoning / IVR fallback)** | **Qwen3-32B on Groq** (`qwen/qwen3-32b`) | $0.29/$0.59 per M tokens, ~535 tok/s, 131k context. Strict schema not needed for free-form reasoning; Qwen3 is plenty. |
| **LLM (fast intent classifier)** | **Rule-based keyword classifier** for IVR menu detection, with a Qwen3-32B fallback | *Pedagogical split.* The whole point of hybrid architecture is learning where the latency budget goes. Using the same LLM for both would erase the lesson. The three-way split (rules → Qwen3 → Kimi K2) exercises the "different models for different jobs" lesson cleanly. |
| **Observability** | **Langfuse** (self-hosted, open-source) starting at **M4** (not M5) | The moment any state machine is in the loop, you need trace IDs linking `{call_sid, turn_index, node, llm_call}`. Stdout logs drown you fast. |
| **Mock payer** | **FastAPI + plain TwiML `<Say>`/`<Gather>`** in M5–M6; upgrade to Pipecat+LLM persona only in M7 if needed | Keeps the mock payer dumb and deterministic while building the real agent. Avoids debugging two voice agents at once. |
| **Runtime** | Python 3.12 + `asyncio` + `uv` for dep mgmt | |

**Realistic cost estimate: $25–$40** (Twilio minutes during debugging + upgrade + TTS during barge-in testing + Groq/Deepgram overage). Not the earlier "under $5."

## Software hygiene baseline

The right habits, installed in M1 and enforced from then on:

- **Tooling:** `ruff` (lint + format), `pyright` in basic mode (not strict — strict on Pipecat internals is a tar pit), `pytest`, `pre-commit` hook running ruff + pyright before every commit.
- **Dependency pinning:** Pipecat is pre-1.0 and breaks weekly. `pyproject.toml` pins exact versions (`pipecat-ai==X.Y.Z`, etc.). `uv.lock` is **committed to git**. No floating ranges.
- **Structured logging from M2, not M4.** Use `structlog` with `contextvars` for `call_sid` and `turn_index`. Langfuse layers on top at M4, not replaces.
- **Secrets:** `.env` is gitignored. `.env.example` is checked in with placeholder values and comments marking which keys are public (Langfuse public key) vs secret (Twilio auth token, Groq key, Cartesia key).
- **Outbound dial allowlist:** `ALLOWED_DESTINATIONS` env var + a hard check in the dialer before every `calls.create()` call. Non-negotiable — a typo in a Twilio number env var should *not* be able to dial a real number. 5 lines of code, lifelong habit.
- **Reproducibility:** `README.md` with setup + run instructions and a `Makefile` with `make agent`, `make mock-payer`, `make test`, `make lint` targets. Both are M1 deliverables.
- **Testing seams:** all external dependencies (LLM, classifier, TTS, ASR) sit behind typed `Protocol` interfaces so unit tests can inject fakes. M3 unit tests must run with **zero network calls**.

**Explicitly cut as YAGNI for this learning project:** GitHub Actions CI (local pre-commit is enough), pyright strict, Docker for the agent itself (Langfuse already needs Docker Compose; don't compound), OpenTelemetry/Prometheus, `tenacity` (hand-write the one retry), multiple env profiles.

## Failure modes

Every external integration needs a defined behavior on failure. No bare `try/except: pass`.

| Integration | Failure | Detection | Behavior |
|---|---|---|---|
| Twilio Media Streams | WebSocket disconnect mid-call | Pipecat transport close event | Log `call_sid` + last state, mark call `terminated_abnormal`, no retry (call is gone) |
| Twilio REST `sendDigits` | 4xx / network error | Exception on API call | Retry once with 500ms backoff; if still failing → transition to `FALLBACK` |
| Deepgram ASR | WebSocket drop mid-utterance | Pipecat `DeepgramSTTService` error frame | Reconnect once automatically; if still failing → `FALLBACK` with logged reason |
| Groq (Qwen3-32B or Kimi K2) | 429 rate limit | HTTP 429 | Exponential backoff, max 2 retries; on timeout (>4s) → `FALLBACK` |
| Groq | 5xx / timeout | HTTP 5xx or 4s timeout | Fail fast → `FALLBACK` (don't chain retries that compound latency) |
| Cartesia TTS | First-byte timeout | 2s timeout | No retry (silence is worse than re-ask); log and short-circuit to a fixed "sorry, could you repeat that" fallback line |
| Mock payer | Number unreachable | Twilio call status `failed`/`busy` | Log and exit; don't retry automatically during development |
| LangGraph node handler | Exception mid-execution | `try/except` in `_run_turn` | Catch in runner, route to `fallback`, log exception with `node` and `call_sid` bound |
| LangGraph routing | Conditional edge returns unknown key | `ValueError` from LangGraph | Catch in runner, route to `fallback`, log with last node name |
| LangGraph recursion | Graph gets stuck (no terminal state) | `recursion_limit=10` set in `ainvoke` config | `GraphRecursionError` → `fallback` with "too many turns" log |
| LangGraph state | Handler returns non-dict | pyright catches at type-check time | Runtime assert as defense in depth; `fallback` on violation |
| GraphRunner | `in_queue` full while pipeline still running | `QueueFull` on `put_nowait` | Drop-oldest policy: discard stale, insert newest |

**Per-call budgets:** Groq timeout = 4s. Cartesia first-byte timeout = 2s. Deepgram utterance-end = configured explicitly in M2 (not defaults).

The `FALLBACK` state is a **safety net** for unexpected failures, not a catch-all for missing error handling. Every transition into `FALLBACK` must log *why*.

## State machine design (LangGraph alongside Pipecat)

Chosen approach: LangGraph for the state machine. The #1 risk is the composition with Pipecat — getting this wrong destroys barge-in latency. Design below is specifically structured to avoid that.

### The non-negotiable rule

**LangGraph must NOT run inside a Pipecat FrameProcessor's frame-handling path.**

If you `await graph.ainvoke(...)` inside `process_frame()`, you block the audio pipeline. A 1.5-second Groq call becomes 1.5 seconds of no TTS cancellation, no VAD handling, no barge-in. Users will feel this as the agent "talking over them" or "freezing."

### The correct shape

LangGraph is an **async state owner** that lives alongside the pipeline, owned by a `GraphRunner` per-call. The Pipecat `FrameProcessor` is a thin adapter that queues transcripts and forwards interrupt events.

**Key realizations (from review):**

- `graph.astream` / `graph.ainvoke` is **one invocation per user turn**, not a full conversation. The per-turn loop lives in the runner, not inside the graph.
- Setting `state["interrupted"] = True` **does not cancel an in-flight LLM call.** Real interrupts require `asyncio.Task.cancel()` on the task running the current turn, which raises `CancelledError` inside `await llm.complete(...)`.
- Handlers must be **cancellation-safe**: on `CancelledError`, abort cleanly and **do not write partial state**.
- Queues are bounded. `in_queue` is drop-oldest (stale transcripts are worthless). `out_queue` blocks (TTS backpressure is real and desired).
- Stale transcripts: on interrupt, cancel in-flight turn, **drain `in_queue` to latest**, start new turn with only the newest transcript.
- **No checkpointer.** `MemorySaver` is cargo culting for this project — you already hold `CallState` in memory in the runner. Re-add only if resume-after-crash becomes a real requirement.
- **`ainvoke` per turn, not `astream`.** The TTS service handles its own streaming. Push the final `response_text` to `out_queue` at turn completion. Simpler, easier to debug.

**GraphRunner shape:**

```python
class GraphRunner:
    def __init__(self, graph: CompiledGraph, call_ctx: CallContext):
        self.graph = graph
        self.call_ctx = call_ctx
        self.in_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=8)
        self.out_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=8)
        self.state: CallState = initial_state(call_ctx)
        self._consumer: asyncio.Task | None = None
        self._current_turn: asyncio.Task | None = None

    async def start(self):
        self._consumer = asyncio.create_task(self._consume())

    async def stop(self):
        if self._current_turn and not self._current_turn.done():
            self._current_turn.cancel()
        if self._consumer:
            self._consumer.cancel()
        # await both, swallow CancelledError

    def mark_interrupted(self):
        """Called from Pipecat thread on VADActiveFrame during TTS."""
        if self._current_turn and not self._current_turn.done():
            self._current_turn.cancel()

    async def _consume(self):
        # Bind contextvars HERE, not in the caller. Contextvars copy at task
        # creation, and call_sid isn't bound until start() is called.
        bind_contextvars(call_sid=self.call_ctx.call_sid, turn_index=0)

        while self.state["current_node"] != "done":
            transcript = await self.in_queue.get()
            # Drain any stale transcripts that piled up during the last turn
            while not self.in_queue.empty():
                transcript = self.in_queue.get_nowait()

            bind_contextvars(turn_index=self.state["turn_count"])
            self._current_turn = asyncio.create_task(
                self._run_turn(transcript)
            )
            try:
                await self._current_turn
            except asyncio.CancelledError:
                log.info("turn_cancelled", reason="interrupt")
                # Don't update state — handler was told to abort cleanly
                continue

    async def _run_turn(self, transcript: str):
        turn_state = {**self.state, "transcript": transcript}
        try:
            result = await self.graph.ainvoke(
                turn_state,
                config={"recursion_limit": 10},
            )
            self.state = result
            self.state["turn_count"] += 1
            if result.get("response_text"):
                await self.out_queue.put(result["response_text"])
        except Exception as exc:
            log.exception("node_error", error=str(exc))
            # Route to fallback on any node-level exception
            self.state["current_node"] = "fallback"
            await self.out_queue.put("Sorry, could you repeat that?")
```

**Pipecat adapter:**

```python
class StateMachineProcessor(FrameProcessor):
    def __init__(self, runner: GraphRunner):
        self.runner = runner

    async def process_frame(self, frame, direction):
        if isinstance(frame, TranscriptionFrame):
            try:
                self.runner.in_queue.put_nowait(frame.text)
            except asyncio.QueueFull:
                # Drop-oldest: discard stale, keep newest
                _ = self.runner.in_queue.get_nowait()
                self.runner.in_queue.put_nowait(frame.text)
        elif isinstance(frame, VADActiveFrame):
            self.runner.mark_interrupted()
        await self.push_frame(frame, direction)
```

**Lifecycle:** one `GraphRunner` per call. Spawned on Pipecat's transport-connect event, stopped on transport-disconnect. Multiple concurrent calls → multiple runners. Never a process-global runner.

### The graph itself

```python
class CallState(TypedDict):
    current_node: str
    patient: PatientInfo
    extracted: Benefits
    turn_count: int
    interrupted: bool
    transcript: str
    response_text: str | None

builder = StateGraph(CallState)
builder.add_node("auth", auth_handler)
builder.add_node("patient_id", patient_handler)
builder.add_node("extract_benefits", extract_handler)
builder.add_node("handoff", handoff_handler)
builder.add_node("fallback", fallback_handler)

builder.add_conditional_edges("auth", route_after_auth, {
    "ok": "patient_id", "retry": "auth", "fail": "fallback"
})
# ...etc
builder.set_entry_point("auth")
```

Each handler is a plain async function that takes `CallState` and returns a partial dict. Handlers call the injected `LLMClient` or `IVRClassifier` — they do NOT import from `pipecat` or `twilio`. This keeps them unit-testable. Dependencies are injected via **closures at graph-build time** (not `functools.partial` — closures are the idiomatic LangGraph pattern, and `partial` breaks once you want per-call `RunnableConfig` injection).

```python
def build_graph(llm: LLMClient, classifier: IVRClassifier) -> CompiledGraph:
    async def extract_handler(state: CallState) -> dict:
        # closure captures llm, classifier
        result = await llm.complete(..., schema=Benefits)
        return {"extracted": result, "current_node": "handoff"}

    builder = StateGraph(CallState)
    builder.add_node("extract_benefits", extract_handler)
    ...
    return builder.compile()  # no checkpointer
```

**Handlers MUST be cancellation-safe.** If `CancelledError` is raised mid-handler (because the user interrupted), the handler must not write partial state to a shared store. Since handlers return dicts rather than mutating inputs, this is mostly automatic — but any external side effect (logging a partial write, sending a webhook) must be inside a `try/finally` that cleans up on cancel.

### Interrupt re-entry

Interrupts are **real task cancellations**, not flag checks:

1. `StateMachineProcessor.process_frame` receives a `VADActiveFrame` → calls `runner.mark_interrupted()`.
2. `mark_interrupted()` calls `self._current_turn.cancel()`.
3. The `await llm.complete(...)` inside the handler raises `CancelledError`.
4. The `_consume()` loop catches it, logs `turn_cancelled`, skips state update, drains stale transcripts, and awaits the next turn.
5. The next handler invocation sees the new transcript; conditional edges route to the resume path based on `current_node`.

The graph itself doesn't need special interrupt nodes. The interrupt is invisible to the graph — from LangGraph's perspective, a turn was simply never completed.

### Testing seams

All external dependencies are injected via typed `Protocol` interfaces so unit tests run with zero network calls:

```python
class LLMClient(Protocol):
    async def complete(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel: ...

class IVRClassifier(Protocol):
    def classify(self, transcript: str) -> ClassifierResult: ...
```

Handlers receive their dependencies via a `functools.partial` or a closure at graph-build time:

```python
def build_graph(llm: LLMClient, classifier: IVRClassifier) -> StateGraph:
    builder.add_node("extract_benefits", partial(extract_handler, llm=llm))
    ...
```

`tests/unit/conftest.py` provides `FakeLLMClient` and `FakeClassifier`. Unit tests call `graph.ainvoke(fake_state)` directly, no Pipecat involved. This is the M3 offline testing strategy.

### Where to go deep (scaffolded)

"Dig into the internals" is only useful if scaffolded. Concrete steps for M3/M4:

1. **Visualize the graph.** After `build_graph()`, call `graph.get_graph().draw_mermaid()` and commit the output to `docs/graph.mmd`. Every time you add/rename a node, re-run it.
2. **Experiment with `astream_events`.** Build a toy 2-node graph in a scratch file. Run `async for ev in graph.astream_events(state, version="v2"): print(ev)`. Observe the event stream: `on_chain_start`, `on_chain_end`, `on_chat_model_start`, etc. This is how Langfuse hooks in at M4.
3. **Set a breakpoint in LangGraph's Pregel runtime.** Break in `langgraph.pregel.Pregel._run_step` (or whatever the current method is called) during a `graph.ainvoke` call. Step through to see how a "superstep" actually executes and how state is merged.
4. **Reading list (in order):** `langgraph/graph/state.py` (how StateGraph compiles), `langgraph/pregel/__init__.py` (the Pregel runtime — supersteps), `Command` primitive docs (for dynamic routing), the `RunnableConfig` contract (how per-call config threads through).
5. **Surprises file.** Maintain `NOTES.md` in the repo. Every time LangGraph does something you didn't expect, write it down with a minimal repro. Three surprises is the M3 exit bar.

## Architecture

```
┌─────────────────────────── VoiceAdmin Agent (caller) ───────────────────────────┐
│                                                                                 │
│   Twilio Media Streams (bidirectional audio WebSocket)                          │
│              │                                                                 │
│              ▼                                                                 │
│   ┌──────────────── Pipecat Pipeline ────────────────┐                         │
│   │                                                  │                         │
│   │   TwilioTransport ──► VAD ──► Deepgram ASR ──►   │                         │
│   │                                                  │                         │
│   │              ┌───────────────────────────────┐   │                         │
│   │              │   State Machine Processor    │   │                         │
│   │              │   (thin Pipecat adapter:     │   │                         │
│   │              │    queues to/from LangGraph  │   │                         │
│   │              │    task running alongside)   │   │                         │
│   │              │                               │   │                         │
│   │              │   Nodes: AUTH → PATIENT_ID    │   │                         │
│   │              │          → EXTRACT_BENEFITS   │   │                         │
│   │              │          → HANDOFF_TO_HUMAN   │   │                         │
│   │              │          → DONE               │   │                         │
│   │              │                               │   │                         │
│   │              │   Fast path: rule-based       │   │                         │
│   │              │   keyword classifier for IVR  │   │                         │
│   │              │   Reasoning: Qwen3-32B for    │   │                         │
│   │              │   clarification + IVR fallback│   │                         │
│   │              │   Extraction: Kimi K2 (strict │   │                         │
│   │              │   JSON schema → Benefits)     │   │                         │
│   │              └───────────────────────────────┘   │                         │
│   │                         │                        │                         │
│   │   ◄── Cartesia TTS ◄── Response Text ◄───────    │                         │
│   │                                                  │                         │
│   └──────────────────────────────────────────────────┘                         │
│                                                                                 │
│   Sideband: Twilio REST API `sendDigits` for DTMF injection                    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                  (phone call)
                                       │
                                       ▼
┌────────────── Mock Payer (second Twilio number) ────────────────┐
│                                                                 │
│   FastAPI webhook returns plain TwiML:                          │
│     - <Gather> for DTMF (fake Aetna IVR menu)                   │
│     - <Say> for prompts                                         │
│     - On handoff: <Say> a scripted rep dialogue or              │
│       (M7 only) switch to Media Streams + LLM persona          │
└─────────────────────────────────────────────────────────────────┘
```

**Execution loop (per turn):**

1. Audio streams in over Twilio WebSocket → Pipecat VAD detects speech.
2. Deepgram transcribes streaming → emits transcript frames.
3. Frame hits the state machine processor. Fast path: regex/keyword classifier checks for IVR menu patterns. If hit, return the DTMF action immediately.
4. Reasoning path: call Qwen3-32B with `{current_node, transcript, extracted_so_far}` for clarification / IVR fallback. For the `extract_benefits` node, call Kimi K2 with strict `json_schema` bound to the `Benefits` Pydantic model.
5. State machine evaluates output → transitions to next node OR fallback.
6. Next node produces a response text → Cartesia TTS → streamed back to Twilio.
7. **Barge-in:** if VAD fires while TTS is playing, Pipecat cancels TTS → ASR picks up the interruption → loop re-enters with "interrupted" state.

## Key files (to create)

```
voiceadmin-learn/
├── README.md                         # Setup, run, troubleshoot
├── Makefile                          # make agent / mock-payer / test / lint
├── pyproject.toml                    # uv-managed deps, pinned exactly
├── uv.lock                           # committed
├── .env.example                      # placeholders; real .env is gitignored
├── .pre-commit-config.yaml           # ruff + pyright
├── agent/
│   ├── main.py                       # Pipecat pipeline + entrypoint
│   ├── graph.py                      # LangGraph StateGraph definition + handlers + Protocols
│   ├── graph_runner.py                # Async task that owns the graph + in/out queues
│   ├── classifier.py                 # Rule-based IVR keyword classifier
│   ├── llm_client.py                 # Groq-backed LLMClient implementation
│   ├── processors/
│   │   └── state_processor.py        # Thin Pipecat FrameProcessor: queues frames to/from graph_runner
│   ├── telephony/
│   │   ├── dialer.py                 # Outbound dial + ALLOWED_DESTINATIONS check
│   │   └── dtmf.py                   # Twilio sendDigits helper
│   ├── prompts/
│   │   └── benefits_extractor.v1.txt # versioned; v2 for M7 experiments
│   ├── logging_config.py             # structlog setup, call_sid/turn_index contextvars
│   ├── config.py                     # typed settings loaded from .env
│   └── schemas.py                    # Pydantic: PatientInfo, Benefits, CallState
├── mock_payer/
│   ├── main.py                       # FastAPI + TwiML webhooks
│   └── ivr_tree.py                   # Deterministic TwiML <Gather> tree
└── tests/
    ├── unit/                         # Zero network. Run on every save.
    │   ├── conftest.py               # FakeLLMClient, FakeClassifier fixtures
    │   ├── test_state_machine.py     # Every transition + fallback path
    │   ├── test_classifier.py
    │   └── test_schemas.py
    └── integration/                  # Mock payer + local Pipecat. No real telephony.
        ├── test_happy_path.py
        ├── test_interruption.py
        └── test_ivr_change.py
```

## Build order (milestones)

1. **M1 — Env, keys, and hygiene baseline.**
   - Sign up: Twilio (**upgrade agent account only off trial, ~$20; mock payer can stay on trial since it only receives inbound calls**), Cartesia, Groq. Deepgram already done.
   - `uv init` + install Pipecat, LangGraph, Groq SDK, Cartesia SDK, Deepgram SDK, FastAPI, structlog, pydantic-settings, python-dotenv (all pinned exactly). Commit `uv.lock`.
   - Install dev deps: `ruff`, `pyright`, `pytest`, `pytest-asyncio`, `pre-commit`. Configure basic ruff + pyright rules; install the pre-commit hook.
   - Write `README.md` with setup instructions and a `Makefile` with `agent`, `mock-payer`, `test`, `lint` targets.
   - `.env.example` with placeholders + comments marking public vs secret keys. Add `.env` and `.env.local` to `.gitignore`.
   - Smoke-test each provider with a one-liner.
   - **Implement the `ALLOWED_DESTINATIONS` allowlist check in a stub `agent/telephony/dialer.py` now** — test it with a fake destination that should be rejected. Build the habit before it matters.

2. **M2 — "Hello world" Pipecat loop + structured logging + quantitative barge-in.**
   - Barebones pipeline: Twilio → ASR → echo back via Cartesia.
   - **Wire `structlog`** with `call_sid` and `turn_index` as contextvars. Every log line is JSON with those fields bound. This is your observability before Langfuse arrives.
   - **Tune explicitly:** Deepgram `endpointing` + `utterance_end_ms`.
   - **Quantitative verification:** log timestamps for `user_started_speaking`, `user_stopped_speaking`, `tts_started`, `tts_cancelled`. Assert cancel-to-silence < 150ms.
   - **False-interrupt test:** play a "cough" during TTS — assert no interrupt.

3. **M3 — LangGraph state machine + GraphRunner (offline, fully unit-testable).**
   - Define `CallState` TypedDict, handlers (as closures), and `build_graph(llm, classifier)` in `agent/graph.py`.
   - Nodes: `auth → patient_id → extract_benefits → handoff → done`, plus `fallback`. Use `add_conditional_edges` for routing.
   - Define `LLMClient` and `IVRClassifier` Protocols; inject via closures at build time.
   - Implement `FakeLLMClient` (with a `slow_mode` flag for cancellation tests) and `FakeClassifier` in `tests/unit/conftest.py`.
   - Implement `GraphRunner` in `agent/graph_runner.py`: per-call lifecycle, bounded queues, per-turn task, `mark_interrupted` that calls `task.cancel()`.
   - Write `tests/unit/test_graph.py`: call `graph.ainvoke(fake_state)` directly. Cover happy path, fallback, malformed LLM JSON, handler exception, routing error, recursion limit.
   - Write `tests/unit/test_graph_runner.py`: (a) runner processes queued transcripts, (b) `mark_interrupted` during a slow turn actually cancels — assert `CancelledError` propagated and state not mutated, (c) stale transcripts dropped after interrupt, (d) `QueueFull` triggers drop-oldest.
   - **M3 unit tests run with zero network calls.** `make test` must pass offline.
   - REPL drive-through with real Groq client as a sanity check.
   - **Go deep:** commit `docs/graph.mmd`, run the `astream_events` experiment, set the Pregel breakpoint, log 3+ surprises in `NOTES.md`.

4. **M4 — Wire GraphRunner into Pipecat + Langfuse.**
   - Implement `state_processor.py`: thin Pipecat FrameProcessor. On `TranscriptionFrame` → `runner.in_queue.put_nowait` with drop-oldest on `QueueFull`. On `VADActiveFrame` → `runner.mark_interrupted()`.
   - Wire `GraphRunner.start()` to Pipecat's transport-connect event, `stop()` to transport-disconnect.
   - Spawn a background task that pumps `runner.out_queue` → `push_frame(TextFrame(...))` for TTS.
   - **Critical verification tests** (before manual phone test):
     - **(a)** Slow-LLM barge-in: inject a 2-second fake LLM, interrupt mid-turn, assert `_current_turn.cancelled()` and no `response_text` emitted.
     - **(b)** Rapid transcripts: push two transcripts 100ms apart, assert only the second reaches a completed turn.
     - **(c)** Handler exception: fake LLM raises, assert state transitions to `fallback` with exception logged.
     - **(d)** Contextvar propagation: every log line from `_consume()` has `call_sid` bound (grep the structlog output in test).
   - Add Langfuse via LangGraph's native tracing integration (env var + callback handler). Verify trace tree shows node transitions and LLM calls.
   - Call the agent from your cell phone, manually role-play the payer IVR. Verify node transitions in Langfuse.

5. **M5 — DTMF spike, then mock payer (IVR leg).**
   - **Spike first:** verify mid-stream DTMF injection actually works. Place a test call into Media Streams, call `sendDigits` on the call SID, confirm the remote side receives DTMF. **Do not build the IVR tree until this works.**
   - Then build mock payer: FastAPI + TwiML `<Gather>` tree. Simple and deterministic.
   - Agent dials mock payer. Happy path: blast through IVR, reach "rep handoff" state.

6. **M6 — Mock payer (scripted human rep leg).**
   - `<Say>` out a fixed rep dialogue after IVR handoff. Scripted, not an LLM.
   - Agent extracts benefits JSON from the scripted conversation.
   - End-to-end happy path should now work.

7. **M7a — Barge-in re-entry.**
   - Mock rep interrupts agent mid-sentence.
   - Assert agent stops, listens, re-enters the state machine cleanly.

8. **M7b — Adaptive IVR (LLM fallback path).**
   - Mock IVR swaps prompt order. Rule-based classifier misses. Qwen3-32B fallback fires, extracts correct action.
   - This is the milestone that exercises the hybrid architecture lesson.

9. **M7c — Ambiguous rep answer (CLARIFICATION path).**
   - Upgrade mock rep to a Pipecat+LLM persona with a quirks toggle. Rep gives ambiguous answer.
   - Agent asks clarifying question, gets structured answer, continues.
   - Only do this if M6 runs cleanly — otherwise cut.

## Reused / referenced libraries

- `pipecat-ai` — audio pipeline, VAD, barge-in, Twilio transport, `DeepgramSTTService`, `CartesiaTTSService`, `GroqLLMService`.
- `langfuse` — observability traces (self-hosted via Docker Compose).
- `pydantic` — `Benefits` schema bound to Kimi K2 via Groq's strict `response_format: json_schema`.
- Pipecat [LangGraph examples](https://github.com/pipecat-ai/pipecat/tree/main/examples) if Option C is chosen.

## Verification

- **M1:** Each provider's smoke-test prints expected output. Twilio upgrade confirmed; 2 numbers purchased. `make lint` and `make test` both pass on empty repo. `ALLOWED_DESTINATIONS` allowlist rejects a fake destination in a unit test. Pre-commit hook blocks a bad commit.
- **M2:** Cancel-to-silence < 150ms on 5 consecutive interruptions. Cough test: 0 false interrupts across 5 trials.
- **M3:** `pytest tests/unit` passes with zero network calls. `graph.ainvoke(fake_state)` lands in `done` with expected `Benefits` on happy path; fallback fires on malformed LLM output; interrupt flag re-routes correctly. REPL drive-through with real Groq lands in `done` with correct `Benefits`. `NOTES.md` has at least 3 LangGraph internals observations.
- **M4:** All four critical verification tests (a–d above) pass. Langfuse UI shows a full trace tree per call with node transitions and LLM calls. Manual call from cell phone completes. Barge-in during a fake 2-second LLM call cancels the turn within 150ms. Every log line from the runner has `call_sid` bound.
- **M5 (spike):** DTMF tones received by a separate test endpoint. IVR tree test: agent completes IVR navigation in 5/5 runs.
- **M6:** Happy-path end-to-end in 5/5 runs. Benefits JSON fully populated.
- **M7a/b/c:** Each edge case passes 5/5 runs. Logs show correct fallback/classifier path fired.
- **Overall success metric:** 5 consecutive successful end-to-end runs on the happy path + M7b adaptive IVR test.

## Decisions made during planning

- **Accounts:** Deepgram exists. M1 includes signup for Twilio (pay-as-you-go, ~$20), Cartesia, Groq.
- **Repo layout:** Single repo, `agent/` and `mock_payer/` subfolders.
- **Observability:** Langfuse, pulled forward from M5 → **M4**.
- **State machine framework:** **LangGraph** — industry-standard library. Runs *alongside* Pipecat via queues, never inside a FrameProcessor's `process_frame()`. User will dig into LangGraph internals whenever something is opaque.
- **Budget:** $25–$40 approved.

## Execution rules (enforced during build)

These rules apply to every milestone and must be added to `CLAUDE.md` before any code is written:

1. **Git worktrees for parallel work.** Each independently executable milestone (or sub-task within a milestone) runs in its own git worktree on its own branch, so multiple streams of work can proceed without stepping on each other. Use `superpowers:using-git-worktrees` to create them.
2. **Parallelize aggressively.** Whenever two or more tasks have no shared state or sequential dependency, dispatch them in parallel via subagents. Use `superpowers:dispatching-parallel-agents` as the default pattern for independent work. Sequential execution is the exception, not the rule.
3. **Simplify before verifying.** At the end of every milestone, run the `superpowers:simplify` skill over the changed code — review for reuse, quality, and efficiency, and fix any issues found. This happens BEFORE verification.
4. **Verify before claiming done.** After simplify, run the `superpowers:verification-before-completion` skill. No milestone is complete, and no commit is made, until verification has produced evidence (test output, lint output, quantitative checks like the barge-in latency assertion). "I think it works" is not verification.
5. **Commit only after simplify + verify both pass.** One commit per completed milestone (or logical sub-task). Commit message describes the milestone completed and the evidence gathered.

## Post-approval actions (in order, before M1 starts)

1. **Update `CLAUDE.md`** with the execution rules above (worktrees, parallelization, simplify, verify, commit discipline) phrased as durable project instructions.
2. **Commit the `CLAUDE.md` change.** This commit is the first action — it encodes the rules before any project code exists.
3. **Create the GitHub repo** (`gh repo create`) for the project and push the initial commit.
4. **Begin M1.**
