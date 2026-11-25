from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from breaker_engine import AgentLoopCircuitBreaker

app = FastAPI(
    title="Agent Loop Circuit Breaker API",
    description="Execution guard for autonomous AI agents that halts infinite tool loops, A-B oscillations, and runaway token/step spend.",
    version="1.0.0",
)

breaker = AgentLoopCircuitBreaker(
    max_identical_calls=3,
    max_total_steps=25,
    max_budget_usd=1.0,
    cost_per_step_usd=0.02,
)


class StepCheckRequest(BaseModel):
    session_id: str = Field(..., min_length=1, example="agent-run-402")
    tool_name: str = Field(..., min_length=1, example="web_search")
    arguments: Dict[str, Any] = Field(default_factory=dict, example={"query": "latest pricing"})
    step_cost_usd: Optional[float] = Field(default=None, ge=0.0)


@app.post("/v1/agent/guard-step", summary="Check & Record Agent Tool Call")
def guard_agent_step(payload: StepCheckRequest):
    try:
        result = breaker.evaluate_step(
            session_id=payload.session_id,
            tool_name=payload.tool_name,
            arguments=payload.arguments,
            step_cost_usd=payload.step_cost_usd,
        )
        return {"status": "success", **result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Circuit breaker error: {str(e)}")


@app.get("/v1/agent/session/{session_id}", summary="Inspect Agent Session Circuit State")
def get_session(session_id: str):
    status = breaker.get_session_status(session_id)
    if not status:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "success", **status}


@app.post("/v1/agent/session/{session_id}/reset", summary="Reset Circuit Breaker for Session")
def reset_session(session_id: str):
    return {"status": "success", **breaker.reset_session(session_id)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8003)
