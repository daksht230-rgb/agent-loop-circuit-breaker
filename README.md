# Agent Loop Circuit Breaker

A stateful **Circuit Breaker & Execution Guard** for autonomous AI agents that detects infinite tool-calling loops, alternating `A -> B -> A -> B` oscillations, and step/cost budget overruns before they burn through API credits.

## Why This Exists
Autonomous agents with tool access frequently get stuck in failure loops—re-invoking the same search or API tool with identical parameters, bouncing back and forth between two tools, or running dozens of expensive steps without converging.

`Agent Loop Circuit Breaker` sits before every tool execution, tracks canonical call fingerprints per session, trips the circuit (`OPEN`) when loop or budget thresholds are breached, and returns an explicit fallback signal.

## Key Capabilities
- **Canonical Call Fingerprinting:** Hashes normalized tool names and sorted arguments to catch repeated calls reliably.
- **Loop & Oscillation Detection:** Trips on repeated identical tool calls and 6-step alternating ping-pong loops (`A -> B -> A -> B`).
- **Step & Cost Budget Enforcement:** Caps total steps and cumulative USD spend per agent session.
- **Session State & Reset Controls:** Inspect or reset circuit state (`CLOSED` / `OPEN`) per `session_id`.

## Quick Start

```bash
pip install -r requirements.txt
uvicorn main:app --port 8003 --reload
curl -X POST [http://127.0.0.1:8003/v1/agent/guard-step](http://127.0.0.1:8003/v1/agent/guard-step) \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "agent-run-101",
    "tool_name": "search_docs",
    "arguments": {"q": "auth token expiry"}
  }'
