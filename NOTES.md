# LangGraph & GraphRunner surprises (M3)

Things that were not obvious from the docs but bit me in M3.

## 1. Cancelling the consumer propagates into the awaited-turn's frame, not into a separate task

When `GraphRunner._consume` is suspended at `await self._current_turn` and
something calls `self._consumer.cancel()`, Python raises `CancelledError` **inside
`_consume` at that `await` line** — not inside `_current_turn`. The turn task
keeps running to completion unless it, too, is cancelled.

The first draft had a single `except asyncio.CancelledError: continue` around the
await, which meant a benign shutdown `cancel()` got mistaken for an interrupt,
the consumer swallowed it, and the loop kept running forever on the next
`in_queue.get()`. Fix: `mark_interrupted()` sets a flag; the except clause
`continue`s only when the flag is set, otherwise re-raises.

## 2. `out_queue.put` happening inside `_run_turn` creates a visible state-read race

Original `_run_turn` did `await out_queue.put(response)` **before returning** to
`_consume`, which then did `self.state = new_state`. Tests that read
`out_queue.get()` and then `runner.state` saw stale state — the producer had
published before the state was committed. Moving the `put` into `_consume`
(after `self.state = new_state`) removes the race: the caller sees the response
only after state is committed, guaranteed by single-threaded asyncio semantics.

General rule: publish-to-observer *must* follow state commit, not precede it.

## 3. LangGraph's `MemorySaver` is a smell, not a feature, for a single process

Every LangGraph tutorial adds `MemorySaver()` to `compile()`. For a single-process
voice agent where state is already held in `GraphRunner.state`, the checkpointer
is pure overhead — it serializes state on every step, adds thread_id plumbing to
`ainvoke` config, and obscures the failure modes (e.g., a stuck call's state now
lives in two places). Dropped it entirely. If we ever need resume-after-crash,
we'll add Postgres-backed checkpointing deliberately — not out of cargo culting.

## 4. TypedDict with `total=False` + pyright basic mode forces `.get()` at every read

`CallState(TypedDict, total=False)` is fine for handler returns (partial
updates), but pyright reports `reportTypedDictNotRequiredAccess` on every
`state["key"]` read. Options: (a) split into a "required" core and an "optional"
tail, (b) use `.get()` everywhere. Picked (b) for M3 — it's uglier at call sites
but keeps the schema single-source-of-truth. May revisit if it gets noisy in M4
when the Pipecat adapter reads from state.

## 5. Dispatcher pattern > chained conditional edges for a "one turn = one ainvoke" loop

First instinct was to wire `auth → patient_id → extract_benefits → …` as a chain
of `add_conditional_edges`, with each handler routing to the next node. That
works for batch execution but breaks the runner model: one `ainvoke` ends up
running *multiple* turns inside a single user-utterance cycle, making interrupt
cancellation semantics fuzzy (which handler was mid-flight when the user talked
over us?).

Dispatcher pattern: entry point is a trivial `dispatcher` node; a conditional
edge reads `state["current_node"]` and routes to exactly one handler node, which
returns `{"current_node": <next_node>, ...}` and goes straight to `END`. Next
`ainvoke` re-enters dispatcher, reads the new `current_node`, routes to the next
handler. Clean 1-to-1 mapping: one user turn → one `ainvoke` → one handler ran.

## 6. TypedDict reducer in LangGraph is replace-per-key, not merge — stale values stick

Found via the M3 REPL drive-through, not by any unit test. `patient_id_handler`
returns `{"current_node": "extract_benefits"}` — no `response_text`. The previous
turn (`auth_handler`) had set `response_text="Calling for member..."`. Expected:
new turn emits nothing. Actual: the runner re-published the auth response, because
LangGraph merges handler returns into the existing state key-by-key, and any key
the handler omits keeps its prior value. That's correct LangGraph semantics — the
error was treating the handler return as the complete turn output.

Fix: `_run_turn` resets `response_text` to `None` in `turn_state` before
`ainvoke`. Now a silent handler produces no response and the out_queue stays
empty. Regression locked in by
`test_silent_handler_does_not_republish_stale_response`.

Meta-lesson: offline unit tests with fake LLMs can miss wire-level composition
bugs — our fakes returned deterministic responses per turn, so the stale-carry
never manifested. The REPL check with a real LLM exposed it immediately.

# M5 surprises

## 5. `log.exception` blocks the event loop ~1.5s on its own when structlog isn't configured

Three pre-existing tests assert behavior that runs through `_run_turn`'s
`except Exception` branch (which calls `log.exception("node_error", ...)`).
When `agent.logging_config.configure_logging()` hasn't been called — which is
the default in tests, since none of them import `agent.main` — structlog falls
back to a Rich-rendered `ConsoleRenderer`. Rendering a deep traceback through
the asyncio + langgraph + pydantic stack takes ~1.5s. That's longer than the
test timeouts (`asyncio.wait_for(out_queue.get(), timeout=1.0)`), so the tests
flake under load and on slower runs.

The naïve fix — call `configure_logging()` from `tests/unit/conftest.py` — runs
into `cache_logger_on_first_use=True`: once the bound logger in
`agent.logging_config.log` is captured, subsequent `structlog.configure()` calls
elsewhere don't propagate. `test_every_log_from_runner_has_call_sid` relies on
reconfiguring structlog at runtime to install a capture processor; adding a
session-level configure in conftest broke that test.

Tests affected (run order matters; the symptom is "1 passed in 0.5s" alone but
"3 failed in ~30s" inside the full suite):
- `test_handler_exception_routes_to_fallback` (in both
  `test_graph_runner.py` and `test_state_processor.py`)
- `test_contextvars_propagate_into_run_turn`

Real-call implication: a handler raising in production blocks the event loop
for ~1.5s during fallback-traceback rendering. Acceptable in the fallback
path, but worth knowing. M5 leaves this as-is; a proper fix is a follow-up.
