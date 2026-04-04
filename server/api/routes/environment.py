from fastapi import APIRouter
from openenv.core.env_server.http_server import StepResponse

from server.core.deps import env_instance
from server.schemas.reset import CustomResetRequest

router = APIRouter()


@router.post("/custom_reset")
def custom_reset(payload: CustomResetRequest):
    """Custom endpoint to inject dynamically generated incident contexts."""
    obs = env_instance.reset(seed=payload.seed)

    # Override the static archetype text with the LLM's dynamic flavor
    env_instance._state.task_description = payload.custom_description

    # Refresh observation to reflect overrides
    obs = env_instance._make_observation(
        message=f"🚨 Custom Incident Registered! {payload.custom_description}", logs=payload.custom_logs, success=True
    )

    return StepResponse(observation=obs.model_dump(), reward=obs.reward, done=obs.done)
