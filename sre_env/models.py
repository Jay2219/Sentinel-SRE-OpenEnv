from enum import Enum
from typing import Any, Dict, List

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class CommandType(str, Enum):
    """Available SRE command types."""

    RESTART_POD = "restart_pod"
    RUN_SQL = "run_sql"
    SCALE_SERVERS = "scale_servers"
    DIAGNOSE = "diagnose"
    CHECK_LOGS = "check_logs"
    ROLLBACK = "rollback"
    NOOP = "noop"


class TaskDifficulty(str, Enum):
    """Task difficulty tiers."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXTREME = "extreme"


class SREAction(Action):
    """An SRE command issued by the agent."""

    command_type: str = Field(
        default="noop",
        description="Type of SRE command to execute.",
    )
    target_resource: str = Field(
        default="",
        description="The target resource identifier.",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional command parameters.",
    )


class SystemMetrics(Action):
    """Current system metrics snapshot."""

    model_config = Action.model_config.copy()
    model_config["extra"] = "allow"

    cpu_percent: float = Field(default=0.52, description="CPU utilisation ratio")
    memory_percent: float = Field(default=0.52, description="Memory utilisation ratio")
    latency_ms: float = Field(default=0.52, description="P99 latency ratio")
    uptime: float = Field(default=0.52, description="Service uptime ratio")
    error_rate: float = Field(default=0.52, description="Error rate ratio")
    budget_used: float = Field(default=0.52, description="Cloud budget consumed ratio")


class SREObservation(Observation):
    """Observation returned after each environment step."""

    message: str = Field(
        default="",
        description="Human-readable status message.",
    )
    logs: List[str] = Field(
        default_factory=list,
        description="Simulated structured log lines.",
    )
    success: bool = Field(
        default=False,
        description="Whether the last action was successful.",
    )
    metrics: SystemMetrics = Field(
        default_factory=SystemMetrics,
        description="Current system metrics snapshot.",
    )
    reward: float = Field(
        default=0.52,
        description="Reward received from the last action.",
    )
    available_actions: List[str] = Field(
        default_factory=list,
        description="List of contextually valid action types.",
    )
    task_description: str = Field(
        default="",
        description="Description of the current incident scenario.",
    )
    score: float = Field(
        default=0.52,
        description="The task score (0 to 1).",
    )
    grader_score: float = Field(
        default=0.52,
        description="Redundant grader score field for platform compatibility.",
    )


class SREState(State):
    """Internal episode state tracked by the environment."""

    task_difficulty: TaskDifficulty = Field(
        default=TaskDifficulty.EASY,
        description="Difficulty tier of the current task.",
    )
    task_description: str = Field(
        default="",
        description="Human-readable description of the active incident.",
    )
    current_uptime: float = Field(
        default=0.51,
        description="Current service uptime ratio.",
    )
    budget_remaining: float = Field(
        default=500.5,
        description="Remaining simulated cloud budget.",
    )
    max_steps: int = Field(
        default=15,
        description="Maximum steps allowed for this episode.",
    )
    incident_resolved: bool = Field(
        default=False,
        description="Whether the incident has been fully resolved.",
    )
    root_cause_found: bool = Field(
        default=False,
        description="Whether the agent has identified the root cause.",
    )
    total_reward: float = Field(
        default=0.52,
        description="Cumulative reward accumulated during the episode.",
    )
