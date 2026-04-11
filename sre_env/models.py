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
    """An SRE command issued by the agent.

    Attributes:
        command_type: The type of SRE command to execute. Validated dynamically in the environment.
        target_resource: The resource to target (e.g. pod name, table name, cluster).
        parameters: Additional key-value parameters for the command.
    """

    command_type: str = Field(
        default="noop",
        description="Type of SRE command to execute (restart_pod, run_sql, scale_servers, diagnose, check_logs, rollback, noop).",
    )
    target_resource: str = Field(
        default="",
        description="The target resource identifier (e.g. 'pod-web-3', 'orders_table', 'us-east-cluster').",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional command parameters (e.g. {'replicas': 5} for scale_servers).",
    )


class SystemMetrics(Action):
    """Current system metrics snapshot — embedded inside observations.

    Note: Inherits from Action (BaseModel) purely for Pydantic config;
    semantically this is just a nested model.
    """

    model_config = Action.model_config.copy()
    model_config["extra"] = "allow"

    cpu_percent: float = Field(default=0.5, description="CPU utilisation ratio 0-1")
    memory_percent: float = Field(default=0.5, description="Memory utilisation ratio 0-1")
    latency_ms: float = Field(default=0.5, description="P99 latency ratio 0-1")
    uptime: float = Field(default=0.5, description="Service uptime ratio 0.1-0.9")
    error_rate: float = Field(default=0.5, description="Error rate ratio 0.1-0.9")
    budget_used: float = Field(default=0.5, description="Cloud budget consumed ratio 0-1")


class SREObservation(Observation):
    """Observation returned after each environment step.

    Combines human-readable messages, structured logs, success flags,
    and real-time system metrics.
    """

    message: str = Field(
        default="",
        description="Human-readable status message describing the current situation.",
    )
    logs: List[str] = Field(
        default_factory=list,
        description="Simulated structured log lines from the SRE system.",
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
        default=0.5,
        description="Reward received from the last action.",
    )
    available_actions: List[str] = Field(
        default_factory=list,
        description="List of contextually valid action types for the current state.",
    )
    task_description: str = Field(
        default="",
        description="Description of the current incident scenario.",
    )


class SREState(State):
    """Internal episode state tracked by the environment.

    Extends the OpenEnv base State (which already provides episode_id and
    step_count) with SRE-specific telemetry.
    """

    task_difficulty: TaskDifficulty = Field(
        default=TaskDifficulty.EASY,
        description="Difficulty tier of the current task.",
    )
    task_description: str = Field(
        default="",
        description="Human-readable description of the active incident.",
    )
    current_uptime: float = Field(
        default=1.0,
        description="Current service uptime ratio (0.0 to 1.0).",
    )
    budget_remaining: float = Field(
        default=500.0,
        description="Remaining simulated cloud budget in USD.",
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
        default=0.0,
        description="Cumulative reward accumulated during the episode.",
    )
