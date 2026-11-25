import json
import time
import hashlib
from typing import Dict, Any, List, Optional


class AgentLoopCircuitBreaker:
    """
    Stateful Circuit Breaker for Autonomous AI Agent Tool Executions.
    Tracks tool-call fingerprints per session/run, detects identical repetitions,
    ping-pong oscillations (A -> B -> A -> B), and step/cost budget breaches,
    and trips the circuit ('OPEN') with a safe fallback action.
    """

    def __init__(
        self,
        max_identical_calls: int = 3,
        max_total_steps: int = 25,
        max_budget_usd: float = 1.0,
        cost_per_step_usd: float = 0.02,
    ):
        if max_identical_calls < 1:
            raise ValueError("max_identical_calls must be at least 1")
        if max_total_steps < 1:
            raise ValueError("max_total_steps must be at least 1")
        if max_budget_usd <= 0:
            raise ValueError("max_budget_usd must be greater than 0")

        self.max_identical_calls = max_identical_calls
        self.max_total_steps = max_total_steps
        self.max_budget_usd = max_budget_usd
        self.cost_per_step_usd = cost_per_step_usd

        # session_id -> session state dictionary
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def _canonical_fingerprint(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        canonical_payload = json.dumps(
            {"tool": tool_name.strip().lower(), "args": arguments},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()[:16]

    def _get_or_create_session(self, session_id: str) -> Dict[str, Any]:
        clean_id = session_id.strip()
        if clean_id not in self._sessions:
            self._sessions[clean_id] = {
                "session_id": clean_id,
                "state": "CLOSED",  # CLOSED (healthy) | OPEN (tripped / blocked)
                "trip_reason": None,
                "steps_executed": 0,
                "spent_usd": 0.0,
                "history": [],  # list of fingerprints
                "fingerprint_counts": {},  # fingerprint -> count
                "created_at": time.time(),
                "updated_at": time.time(),
            }
        return self._sessions[clean_id]

    def _detect_ping_pong_loop(self, history: List[str]) -> bool:
        """Detects 2-step alternating loops like [A, B, A, B, A, B] (6 recent calls)."""
        if len(history) < 6:
            return False
        recent = history[-6:]
        return (
            recent[0] == recent[2] == recent[4]
            and recent[1] == recent[3] == recent[5]
            and recent[0] != recent[1]
        )

    def evaluate_step(
        self,
        session_id: str,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        step_cost_usd: Optional[float] = None,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()

        if not session_id or not session_id.strip():
            raise ValueError("session_id cannot be empty")
        if not tool_name or not tool_name.strip():
            raise ValueError("tool_name cannot be empty")

        session = self._get_or_create_session(session_id)
        args_dict = arguments if arguments is not None else {}
        cost = self.cost_per_step_usd if step_cost_usd is None else max(0.0, step_cost_usd)

        # If circuit is already OPEN, block immediately without incrementing spend
        if session["state"] == "OPEN":
            return {
                "allowed": False,
                "circuit_state": "OPEN",
                "session_id": session["session_id"],
                "trip_reason": session["trip_reason"],
                "fallback_action": "HALT_AGENT_AND_RETURN_PARTIAL_STATE",
                "steps_executed": session["steps_executed"],
                "max_total_steps": self.max_total_steps,
                "spent_usd": round(session["spent_usd"], 6),
                "max_budget_usd": self.max_budget_usd,
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 3),
            }

        fp = self._canonical_fingerprint(tool_name, args_dict)
        next_steps = session["steps_executed"] + 1
        next_spend = session["spent_usd"] + cost
        next_fp_count = session["fingerprint_counts"].get(fp, 0) + 1
        prospective_history = session["history"] + [fp]

        trip_reason: Optional[str] = None

        # 1. Check identical repeated tool-call loop
        if next_fp_count > self.max_identical_calls:
            trip_reason = (
                f"identical_tool_loop_detected: '{tool_name}' called {next_fp_count} times "
                f"with same arguments (limit: {self.max_identical_calls})"
            )

        # 2. Check alternating ping-pong loop (A -> B -> A -> B)
        elif self._detect_ping_pong_loop(prospective_history):
            trip_reason = "oscillation_loop_detected: alternating A-B-A-B tool call pattern"

        # 3. Check max step count
        elif next_steps > self.max_total_steps:
            trip_reason = f"max_steps_exceeded: {next_steps} > {self.max_total_steps}"

        # 4. Check budget cap
        elif round(next_spend, 8) > round(self.max_budget_usd, 8):
            trip_reason = (
                f"budget_cap_exceeded: ${round(next_spend, 4)} > ${round(self.max_budget_usd, 4)}"
            )

        session["updated_at"] = time.time()

        if trip_reason is not None:
            session["state"] = "OPEN"
            session["trip_reason"] = trip_reason
            return {
                "allowed": False,
                "circuit_state": "OPEN",
                "session_id": session["session_id"],
                "trip_reason": trip_reason,
                "fallback_action": "ESCALATE_TO_HUMAN_OR_SYNTHESIZE_ANSWER",
                "steps_executed": session["steps_executed"],
                "max_total_steps": self.max_total_steps,
                "spent_usd": round(session["spent_usd"], 6),
                "max_budget_usd": self.max_budget_usd,
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 3),
            }

        # Step is safe: commit state updates
        session["steps_executed"] = next_steps
        session["spent_usd"] = next_spend
        session["fingerprint_counts"][fp] = next_fp_count
        session["history"] = prospective_history

        return {
            "allowed": True,
            "circuit_state": "CLOSED",
            "session_id": session["session_id"],
            "fingerprint": fp,
            "identical_call_count": next_fp_count,
            "steps_executed": session["steps_executed"],
            "max_total_steps": self.max_total_steps,
            "spent_usd": round(session["spent_usd"], 6),
            "max_budget_usd": self.max_budget_usd,
            "latency_ms": round((time.perf_counter() - start_time) * 1000, 3),
        }

    def reset_session(self, session_id: str) -> Dict[str, Any]:
        clean_id = session_id.strip()
        if clean_id in self._sessions:
            del self._sessions[clean_id]
        fresh = self._get_or_create_session(clean_id)
        return {
            "session_id": fresh["session_id"],
            "circuit_state": fresh["state"],
            "status": "reset",
        }

    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        clean_id = session_id.strip()
        session = self._sessions.get(clean_id)
        if not session:
            return None
        return {
            "session_id": session["session_id"],
            "circuit_state": session["state"],
            "trip_reason": session["trip_reason"],
            "steps_executed": session["steps_executed"],
            "max_total_steps": self.max_total_steps,
            "spent_usd": round(session["spent_usd"], 6),
            "max_budget_usd": self.max_budget_usd,
            "unique_call_fingerprints": len(session["fingerprint_counts"]),
        }
