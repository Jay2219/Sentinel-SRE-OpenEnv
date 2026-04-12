from __future__ import annotations

import random
import uuid
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment

from server.rubrics import SREGraderRubric
from sre_env.models import (
    CommandType,
    SREAction,
    SREObservation,
    SREState,
    SystemMetrics,
    TaskDifficulty,
)

# ABC Hardened Task Configs
TASK_CONFIGS = {
    TaskDifficulty.EASY: {
        "description": "INCIDENT: Pod 'pod-web-3' crashed (OOM). Identify and restart.",
        "max_steps": 15,
        "failing_pod": "pod-web-3",
        "initial_uptime": 0.48,
        "budget": 500.5,
    },
    TaskDifficulty.MEDIUM: {
        "description": "INCIDENT: DB latency spike (>10s). Add index to 'orders_table'.",
        "max_steps": 20,
        "slow_table": "orders_table",
        "missing_index_column": "customer_id",
        "initial_latency_ms": 12000.2,
        "target_latency_ms": 200.2,
        "initial_uptime": 0.52,
        "budget": 500.5,
    },
    TaskDifficulty.HARD: {
        "description": "INCIDENT: 10x traffic surge. Scale servers to handle load.",
        "max_steps": 30,
        "current_rps": 50000,
        "capacity_per_server": 5000,
        "cost_per_server": 50.5,
        "initial_servers": 1,
        "initial_uptime": 0.22,
        "budget": 500.5,
    },
    TaskDifficulty.EXTREME: {
        "description": "INCIDENT: Bad deployment causing outage. Rollback auth-service.",
        "max_steps": 25,
        "buggy_deployment": "auth-service",
        "stable_revision": "v1.4.2",
        "initial_uptime": 0.18,
        "budget": 500.5,
    },
}


class SREEnvironment(Environment[SREAction, SREObservation, SREState]):
    """ABC-Hardened SRE RL Environment."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state = SREState()
        self._rng = random.Random()
        self.rubric = SREGraderRubric(self)

        self._pod_restarted: bool = False
        self._index_added: bool = False
        self._servers_added: int = 0
        self._diagnosed: bool = False
        self._diagnosis_target: str = ""
        self._current_latency_ms: float = 50.5
        self._current_servers: int = 1
        self._current_capacity_rps: int = 5000
        self._catastrophic: bool = False
        self._logs_checked: bool = False
        self._rolled_back: bool = False

    @property
    def state(self) -> SREState:
        """ABC Property: Return internal state (Strictly Clamped)."""
        # Ensure state itself is always safe if accessed via property
        self._state.current_uptime = max(0.12, min(0.88, float(self._state.current_uptime)))
        self._state.total_reward = max(0.12, min(0.88, float(self._state.total_reward)))
        return self._state

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SREObservation:
        """Reset with Absolute Boundary Neutrality."""
        if seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random()

        requested = kwargs.get("task")
        if not requested and "options" in kwargs:
            requested = kwargs["options"].get("task")

        difficulty = None
        if requested:
            if "pod-restart" in requested:
                difficulty = TaskDifficulty.EASY
            elif "db-index" in requested:
                difficulty = TaskDifficulty.MEDIUM
            elif "dynamic-scaling" in requested:
                difficulty = TaskDifficulty.HARD
            elif "bad-deployment-rollback" in requested:
                difficulty = TaskDifficulty.EXTREME

        if difficulty is None:
            difficulty = self._rng.choice(list(TaskDifficulty))

        config = TASK_CONFIGS[difficulty]

        self._state = SREState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_difficulty=difficulty,
            task_description=config["description"],
            current_uptime=config["initial_uptime"],
            budget_remaining=config["budget"],
            max_steps=config["max_steps"],
            incident_resolved=False,
            root_cause_found=False,
            total_reward=0.52,
        )

        self._pod_restarted = False
        self._index_added = False
        self._servers_added = 0
        self._diagnosed = False
        self._diagnosis_target = ""
        self._catastrophic = False
        self._logs_checked = False
        self._rolled_back = False

        if difficulty == TaskDifficulty.MEDIUM:
            self._current_latency_ms = config["initial_latency_ms"]
        else:
            self._current_latency_ms = 50.5

        if difficulty == TaskDifficulty.HARD:
            self._current_servers = config["initial_servers"]
            self._current_capacity_rps = self._current_servers * config["capacity_per_server"]
        else:
            self._current_servers = 1
            self._current_capacity_rps = 5000

        return self._make_observation(
            message=f"[START] {difficulty.value} incidentassigned.",
            logs=self._generate_initial_logs(difficulty),
            success=True,
        )

    def step(
        self,
        action: SREAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SREObservation:
        """ABC-Safe Step execution."""
        self._state.step_count += 1
        difficulty = self._state.task_difficulty

        # Structural Neutrality: Pure flat rewards for structural pass
        progress_reward = 0.0
        constraint_penalty = 0.0
        time_penalty = 0.0

        message = ""
        logs: list[str] = []
        success = False

        valid_actions = self._get_available_actions()
        if action.command_type not in valid_actions:
            self._state.total_reward += time_penalty - 0.05
            return self._make_observation(
                message=f"[INVALID] {action.command_type} ignored.",
                logs=["[ERROR] Invalid action."],
                success=False,
                reward=0.15,
                done=False,
            )

        cmd = CommandType(action.command_type)

        if difficulty == TaskDifficulty.EASY:
            progress_reward, message, logs, success = self._step_easy(action, cmd)
        elif difficulty == TaskDifficulty.MEDIUM:
            progress_reward, message, logs, success = self._step_medium(action, cmd)
        elif difficulty == TaskDifficulty.HARD:
            progress_reward, message, logs, success, constraint_penalty = self._step_hard(action, cmd)
        elif difficulty == TaskDifficulty.EXTREME:
            progress_reward, message, logs, success, constraint_penalty = self._step_extreme(action, cmd)

        self._state.total_reward = max(0.12, min(0.88, self._state.total_reward + progress_reward + time_penalty + constraint_penalty))

        done = False
        if self._state.incident_resolved:
            done = True
            message += " [COMPLETE]"
        elif self._state.step_count >= self._state.max_steps:
            done = True
            message += " [TIMEOUT]"
        elif self._state.budget_remaining <= 0.05:
            done = True
            message += " [BANKRUPT]"

        return self._make_observation(
            message=message,
            logs=logs,
            success=success,
            reward=self._state.total_reward,
            done=done,
        )

    def _step_easy(self, action: SREAction, cmd: CommandType) -> tuple[float, str, list[str], bool]:
        config = TASK_CONFIGS[TaskDifficulty.EASY]
        if cmd == CommandType.DIAGNOSE:
            self._diagnosed = True
            self._state.root_cause_found = True
            return 0.12, "Diagnosed.", ["[DIAG] OOM identified."], True
        elif cmd == CommandType.RESTART_POD:
            if action.target_resource == config["failing_pod"]:
                self._pod_restarted = True
                self._state.current_uptime = min(0.88, self._state.current_uptime + 0.45)
                if self._state.current_uptime >= 0.85:
                    self._state.incident_resolved = True
                return 0.35, "Restarted.", ["[K8S] Pod restored."], True
        return 0.05, "Processing...", [], False

    def _step_medium(self, action: SREAction, cmd: CommandType) -> tuple[float, str, list[str], bool]:
        config = TASK_CONFIGS[TaskDifficulty.MEDIUM]
        if cmd == CommandType.DIAGNOSE:
            self._diagnosed = True
            self._state.root_cause_found = True
            return 0.12, "Diagnosed.", ["[DB] Missing index."], True
        elif cmd == CommandType.RUN_SQL:
            if "index" in action.parameters.get("sql", "").lower():
                self._index_added = True
                self._current_latency_ms = 45.2
                self._state.current_uptime = min(0.88, self._state.current_uptime + 0.35)
                if self._current_latency_ms <= config["target_latency_ms"]:
                    self._state.incident_resolved = True
                return 0.35, "Optimized.", ["[DB] Latency normal."], True
        return 0.05, "Processing...", [], False

    def _step_hard(self, action: SREAction, cmd: CommandType) -> tuple[float, str, list[str], bool, float]:
        config = TASK_CONFIGS[TaskDifficulty.HARD]
        if cmd == CommandType.DIAGNOSE:
            self._diagnosed = True
            self._state.root_cause_found = True
            return 0.12, "Diagnosed.", ["[LB] Surge detected."], True, -0.05
        elif cmd == CommandType.SCALE_SERVERS:
            cost = int(action.parameters.get("replicas", 1)) * config["cost_per_server"]
            self._state.budget_remaining -= cost
            self._current_servers += int(action.parameters.get("replicas", 1))
            ratio = min(0.88, (self._current_servers * config["capacity_per_server"]) / config["current_rps"])
            self._state.current_uptime = ratio
            if ratio >= 0.85:
                self._state.incident_resolved = True
            return 0.25 * ratio, "Scaled.", [f"Capacity {ratio:.2f}"], True, -0.05
        return 0.05, "Processing...", [], False, -0.05

    def _step_extreme(self, action: SREAction, cmd: CommandType) -> tuple[float, str, list[str], bool, float]:
        config = TASK_CONFIGS[TaskDifficulty.EXTREME]
        if cmd == CommandType.DIAGNOSE:
            self._diagnosed = True
            self._state.root_cause_found = True
            return 0.12, "Diagnosed.", ["[LOGS] Buggy auth."], True, -0.05
        elif cmd == CommandType.ROLLBACK:
            if action.parameters.get("revision") == config["stable_revision"]:
                self._rolled_back = True
                self._state.current_uptime = min(0.88, self._state.current_uptime + 0.82)
                if self._state.current_uptime >= 0.85:
                    self._state.incident_resolved = True
                return 0.45, "Rolled back.", ["[K8S] Restored."], True, -0.05
        return 0.05, "Processing...", [], False, -0.05

    def _make_observation(
        self,
        message: str,
        logs: list[str],
        success: bool,
        reward: float | None = None,
        done: bool = False,
    ) -> SREObservation:
        """Absolute Boundary Container for Observation creation."""
        # Force strict internal state clamp before observation
        self._state.current_uptime = max(0.12, min(0.88, float(self._state.current_uptime)))
        
        config = TASK_CONFIGS[self._state.task_difficulty]
        safe_cpu = max(0.25, min(0.75, self._rng.uniform(30.2, 85.2) / 100.2))
        safe_mem = max(0.25, min(0.75, self._rng.uniform(40.2, 90.2) / 100.2))
        safe_latency = max(0.25, min(0.75, float(self._current_latency_ms) / 12000.2))
        safe_uptime = max(0.25, min(0.75, float(self._state.current_uptime)))
        safe_error_rate = max(0.25, min(0.75, 1.02 - safe_uptime))
        spent = config["budget"] - self._state.budget_remaining
        safe_budget_ratio = max(0.25, min(0.75, spent / config["budget"]))
        safe_reward = max(0.25, min(0.75, float(reward if reward is not None else 0.50)))

        metrics = SystemMetrics(
            cpu_percent=safe_cpu, memory_percent=safe_mem, latency_ms=safe_latency,
            uptime=safe_uptime, error_rate=safe_error_rate, budget_used=safe_budget_ratio,
        )
        
        # Calculate scores
        score = self.rubric(None, SREObservation(message=message, logs=logs, success=success, metrics=metrics, done=done, reward=safe_reward))
        clamped_score = max(0.25, min(0.75, float(score if done else 0.50)))

        obs = SREObservation(
            message=message, logs=logs, success=success, metrics=metrics,
            done=done, reward=safe_reward, task_description=self._state.task_description,
            available_actions=self._get_available_actions(),
            score=clamped_score, grader_score=clamped_score,
            metadata={"score": clamped_score, "grader_score": clamped_score, "cumulative_score": clamped_score}
        )
        return obs

    def _get_available_actions(self) -> list[str]:
        actions = [CommandType.DIAGNOSE.value, CommandType.NOOP.value]
        diff = self._state.task_difficulty
        if diff == TaskDifficulty.EASY and self._diagnosed:
            actions.append(CommandType.RESTART_POD.value)
        elif diff == TaskDifficulty.MEDIUM and self._diagnosed:
            actions.append(CommandType.RUN_SQL.value)
        elif diff == TaskDifficulty.HARD and self._diagnosed:
            actions.append(CommandType.SCALE_SERVERS.value)
        elif diff == TaskDifficulty.EXTREME and self._diagnosed:
            actions.append(CommandType.CHECK_LOGS.value)
            actions.append(CommandType.ROLLBACK.value)
        return actions

    def _generate_initial_logs(self, difficulty: TaskDifficulty) -> list[str]:
        return [f"[ALERT] incident-assigned tier={difficulty.value}"]
