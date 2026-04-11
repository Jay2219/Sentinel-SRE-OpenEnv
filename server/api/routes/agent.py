import asyncio
import json
import os
import re
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from server.core.deps import env_instance
from sre_env.constants.prompts import SYSTEM_PROMPT
from sre_env.models import SREAction

load_dotenv()

router = APIRouter()

MAX_STEPS = 15
_executor = ThreadPoolExecutor(max_workers=2)


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from LLM output."""
    try:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            clean = match.group(1).replace("```json", "").replace("```", "")
            return json.loads(clean)
        return json.loads(text)
    except Exception:
        return {"command_type": "noop", "target_resource": "", "parameters": {}}


def _sse_event(event: str, data: dict) -> str:
    """Format a Server-Sent Event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _run_agent(seed: int, custom_desc: str | None = None) -> AsyncGenerator[str, None]:
    """Generator that runs the full agent episode and yields SSE events."""

    api_base = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME")
    hf_token = os.getenv("HF_TOKEN")

    if not api_base or not model_name:
        yield _sse_event(
            "log",
            {
                "type": "alert",
                "text": "[ERROR] API_BASE_URL or MODEL_NAME not configured. Set them in HF Space Secrets.",
            },
        )
        yield _sse_event("done", {"score": "N/A", "steps": 0})
        return

    from openai import OpenAI

    client = OpenAI(base_url=api_base, api_key=hf_token)

    # --- Reset environment ---
    yield _sse_event("log", {"type": "system", "text": f"[SYSTEM] Resetting environment (seed={seed})..."})
    obs = env_instance.reset(seed=seed)

    if custom_desc:
        env_instance._state.task_description = custom_desc
        obs = env_instance._make_observation(
            message=f"🚨 Custom Incident: {custom_desc}",
            logs=[],
            success=True,
        )

    obs_dict = obs.model_dump()
    task = obs_dict.get("task_description", "Unknown")
    yield _sse_event("task", {"description": task})
    yield _sse_event("log", {"type": "system", "text": f"[INCIDENT] {task[:120]}"})

    for log_line in obs_dict.get("logs", []):
        yield _sse_event("log", {"type": "system", "text": log_line})

    # --- Stats ---
    yield _sse_event(
        "stats",
        {
            "uptime": f"{env_instance._state.current_uptime * 100:.0f}%",
            "budget": f"${env_instance._state.budget_remaining:.0f}",
        },
    )

    await asyncio.sleep(0.1)

    # --- Agent loop ---
    done = obs_dict.get("done", False)
    total_reward = 0.45

    for step in range(1, MAX_STEPS + 1):
        if done:
            break

        yield _sse_event("log", {"type": "system", "text": f"── Step {step} ──────────────────────────"})

        # Call LLM (run in thread so we can yield heartbeats)
        yield _sse_event("log", {"type": "system", "text": "[AGENT] Thinking..."})
        yield ": heartbeat\n\n"  # SSE comment keeps HF proxy alive

        try:
            history = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context: {json.dumps(obs_dict)}"},
            ]
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                _executor,
                lambda: client.chat.completions.create(
                    model=model_name,
                    messages=history,
                    max_tokens=512,
                    temperature=0.1,
                    timeout=30,
                ),
            )
            raw = response.choices[0].message.content.strip()
            action_dict = _extract_json(raw)
        except Exception as e:
            yield _sse_event("log", {"type": "alert", "text": f"[LLM ERROR] {e}"})
            break

        cmd = action_dict.get("command_type", "noop")
        target = action_dict.get("target_resource", "")
        yield _sse_event("log", {"type": "user", "text": f"[ACTION] {cmd} on {target}"})

        # Step environment
        try:
            action = SREAction(**action_dict)
            obs = env_instance.step(action)
            obs_dict = obs.model_dump()
        except Exception as e:
            yield _sse_event("log", {"type": "alert", "text": f"[ENV ERROR] {e}"})
            break

        reward = obs_dict.get("reward", 0.45) or 0.45
        total_reward += reward
        done = obs_dict.get("done", False)
        msg = obs_dict.get("message", "")

        log_type = "system"
        if "Success" in msg or "restored" in msg:
            log_type = "system"
        elif "⚠️" in msg or "penalty" in msg.lower():
            log_type = "alert"

        yield _sse_event("log", {"type": log_type, "text": f"[RESULT] {msg}"})
        yield _sse_event("log", {"type": "system", "text": f"[REWARD] {reward:+.3f} | Total: {total_reward:+.3f}"})

        for log_line in obs_dict.get("logs", []):
            yield _sse_event("log", {"type": "system", "text": log_line})

        yield _sse_event(
            "stats",
            {
                "uptime": f"{env_instance._state.current_uptime * 100:.0f}%",
                "budget": f"${env_instance._state.budget_remaining:.0f}",
            },
        )

        await asyncio.sleep(0.1)

    # --- Final score ---
    score = obs_dict.get("metadata", {}).get("grader_score", "N/A")
    final_msg = obs_dict.get("message", "")
    if "GRADER_SCORE:" in final_msg:
        try:
            score = final_msg.split("GRADER_SCORE: ")[1].split("]")[0]
        except Exception:
            pass

    yield _sse_event("log", {"type": "system", "text": "━━━ Episode Complete ━━━"})
    yield _sse_event("log", {"type": "user", "text": f"[GRADE] Final Score: {score}"})
    yield _sse_event("done", {"score": str(score), "steps": step})


@router.get("/api/run_agent")
async def run_agent(
    seed: int = Query(default=2, description="Environment seed"),
    custom_desc: str | None = Query(default=None, description="Custom incident description"),
):
    """Stream the agent's actions via Server-Sent Events."""
    return StreamingResponse(
        _run_agent(seed, custom_desc),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
