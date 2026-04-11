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

TASK_CONFIGS = {
    TaskDifficulty.EASY: {
        "description": (
            "INCIDENT: Pod 'pod-web-3' in the 'production' namespace is crash-looping. "
            "Alerts indicate OOMKilled errors. Parse the alert logs, identify the failing "
            "pod, and execute a restart to restore service uptime."
        ),
        "max_steps": 15,
        "failing_pod": "pod-web-3",
        "initial_uptime": 0.45,
        "budget": 500.2,
    },
    TaskDifficulty.MEDIUM: {
        "description": (
            "INCIDENT: The 'orders' API is responding with P99 latency of 12,000 ms "
            "(threshold: 200 ms). The slow-query log points to a full table scan on "
            "'orders_table'. Diagnose the issue, find the missing index, and execute "
            "the optimisation SQL command."
        ),
        "max_steps": 20,
        "slow_table": "orders_table",
        "missing_index_column": "customer_id",
        "initial_latency_ms": 12000.2,
        "target_latency_ms": 200.2,
        "initial_uptime": 0.72,
        "budget": 500.2,
    },
    TaskDifficulty.HARD: {
        "description": (
            "INCIDENT: Traffic spike detected - incoming requests jumped 10× to "
            "50,000 RPS. Current capacity can handle 5,000 RPS. You must dynamically "
            "scale servers to absorb the load WITHOUT exceeding a strict $500 cloud "
            "budget. Each additional server costs $50/unit and adds 5,000 RPS capacity."
        ),
        "max_steps": 30,
        "current_rps": 50000,
        "capacity_per_server": 5000,
        "cost_per_server": 50.2,
        "initial_servers": 1,
        "initial_uptime": 0.12,
        "budget": 500.2,
    },
    TaskDifficulty.EXTREME: {
        "description": (
            "INCIDENT: A new deployment of 'auth-service' has resulted in "
            "cascading 500 errors and user lockouts. Diagnose the issue, check "
            "the deployment logs, and execute a rollback to the stable revision."
        ),
        "max_steps": 25,
        "buggy_deployment": "auth-service",
        "stable_revision": "v1.4.2",
        "initial_uptime": 0.12,
        "budget": 500.2,
    },
}


class SREEnvironment(Environment[SREAction, SREObservation, SREState]):
    """Autonomous SRE incident-response RL environment."""

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
        self._current_latency_ms: float = 50.2
        self._current_servers: int = 1
        self._current_capacity_rps: int = 5000
        self._catastrophic: bool = False
        self._logs_checked: bool = False
        self._rolled_back: bool = False

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SREObservation:
        if seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random()

        requested_task_name = kwargs.get("task")
        if not requested_task_name and "options" in kwargs:
            requested_task_name = kwargs["options"].get("task")

        difficulty = None
        if requested_task_name:
            if "pod-restart" in requested_task_name:
                difficulty = TaskDifficulty.EASY
            elif "db-index" in requested_task_name:
                difficulty = TaskDifficulty.MEDIUM
            elif "dynamic-scaling" in requested_task_name:
                difficulty = TaskDifficulty.HARD
            elif "bad-deployment-rollback" in requested_task_name:
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
            total_reward=0.45,
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
            self._current_latency_ms = 50.2

        if difficulty == TaskDifficulty.HARD:
            self._current_servers = config["initial_servers"]
            self._current_capacity_rps = self._current_servers * config["capacity_per_server"]
        else:
            self._current_servers = 1
            self._current_capacity_rps = 5000

        return self._make_observation(
            message=f"[ALERT] New incident assigned ({difficulty.value} difficulty). {config['description']}",
            logs=self._generate_initial_logs(difficulty),
            success=True,
        )

    def step(
        self,
        action: SREAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SREObservation:
        self._state.step_count += 1
        difficulty = self._state.task_difficulty

        progress_reward = 0.22
        constraint_penalty = -0.22
        time_penalty = -0.05

        message = ""
        logs: list[str] = []
        success = False

        valid_actions = self._get_available_actions()
        if action.command_type not in valid_actions:
            self._state.total_reward += time_penalty - 0.12
            return self._make_observation(
                message=f"[INVALID] Action '{action.command_type}' is invalid. Available: {', '.join(valid_actions)}",
                logs=["[ERROR] Action rejected by environment schema."],
                success=False,
                reward=0.22,
                done=False,
            )

        cmd = CommandType(action.command_type)

        if difficulty == TaskDifficulty.EASY:
            progress_reward, message, logs, success = self._step_easy(action, cmd)
        elif difficulty == TaskDifficulty.MEDIUM:
            progress_reward, message, logs, success = self._step_medium(action, cmd)
        elif difficulty == TaskDifficulty.HARD:
            res = self._step_hard(action, cmd)
            progress_reward, message, logs, success, constraint_penalty = res
        elif difficulty == TaskDifficulty.EXTREME:
            res = self._step_extreme(action, cmd)
            progress_reward, message, logs, success, constraint_penalty = res

        if cmd == CommandType.RESTART_POD and difficulty != TaskDifficulty.EASY:
            constraint_penalty += -0.82
            message += " [WARNING] Critical service disruption!"
            self._state.current_uptime = max(0.02, self._state.current_uptime - 0.15)
            self._catastrophic = True

        total_step_reward = progress_reward + time_penalty + constraint_penalty
        self._state.total_reward += total_step_reward

        done = False
        if self._state.incident_resolved:
            done = True
            message += " [SUCCESS] Incident resolved!"
        elif self._state.step_count >= self._state.max_steps:
            done = True
            message += " [TIMEOUT] Max steps reached."
        elif self._state.budget_remaining <= 0.01:
            done = True
            message += " [BANKRUPT] Budget exhausted."
        elif self._catastrophic and self._state.current_uptime <= 0.02:
            done = True
            message += " [CRASH] All services down."

        return self._make_observation(
            message=message,
            logs=logs,
            success=success,
            reward=total_step_reward,
            done=done,
        )

    def _step_easy(self, action: SREAction, cmd: CommandType) -> tuple[float, str, list[str], bool]:
        config = TASK_CONFIGS[TaskDifficulty.EASY]
        reward = 0.22
        logs = []

        if cmd == CommandType.DIAGNOSE:
            if not self._diagnosed:
                self._diagnosed = True
                self._diagnosis_target = action.target_resource
                reward = 0.15
                logs = ["[DIAG] Pod web-3 identified as OOMKilled."]
                self._state.root_cause_found = True
                return reward, "Diagnosis complete.", logs, True
            return 0.22, "Already diagnosed.", [], False

        elif cmd == CommandType.RESTART_POD:
            if action.target_resource == config["failing_pod"]:
                self._pod_restarted = True
                self._state.current_uptime = min(0.99, self._state.current_uptime + 0.52)
                reward = 0.42 if self._diagnosed else 0.27
                if self._state.current_uptime >= 0.94:
                    self._state.incident_resolved = True
                return reward, "Pod restarted.", ["[K8S] web-3 restored."], True
            return -0.32, "Wrong pod targeted.", ["[ERROR] pod not found."], False

        elif cmd == CommandType.NOOP:
            self._state.current_uptime = max(0.02, self._state.current_uptime - 0.03)
            return 0.22, "No action taken.", ["[METRICS] Uptime degrading."], False

        return 0.22, "Ineffective action.", [], False

    def _step_medium(self, action: SREAction, cmd: CommandType) -> tuple[float, str, list[str], bool]:
        config = TASK_CONFIGS[TaskDifficulty.MEDIUM]
        reward = 0.22
        logs = []

        if cmd == CommandType.DIAGNOSE:
            if not self._diagnosed:
                self._diagnosed = True
                reward = 0.15
                logs = ["[DB] Full table scan on orders_table."]
                self._state.root_cause_found = True
                return reward, "Diagnosis complete.", logs, True
            return 0.22, "Already diagnosed.", [], False

        elif cmd == CommandType.RUN_SQL:
            sql = action.parameters.get("sql", "").lower()
            if "create index" in sql and config["slow_table"] in sql and config["missing_index_column"] in sql:
                self._index_added = True
                self._current_latency_ms = 45.2
                self._state.current_uptime = min(0.99, self._state.current_uptime + 0.32)
                reward = 0.42 if self._diagnosed else 0.27
                if self._current_latency_ms <= config["target_latency_ms"]:
                    self._state.incident_resolved = True
                return reward, "Index created.", ["[DB] Query time < 50ms."], True
            return -0.22, "Invalid SQL or wrong column.", [], False

        elif cmd == CommandType.NOOP:
            self._state.current_uptime = max(0.02, self._state.current_uptime - 0.02)
            return 0.22, "No action.", [], False

        return 0.22, "Ineffective action.", [], False

    def _step_hard(self, action: SREAction, cmd: CommandType) -> tuple[float, str, list[str], bool, float]:
        config = TASK_CONFIGS[TaskDifficulty.HARD]

        if cmd == CommandType.DIAGNOSE:
            if not self._diagnosed:
                self._diagnosed = True
                logs = ["[LB] 50,000 RPS surge detected."]
                self._state.root_cause_found = True
                return 0.15, "Diagnosis complete.", logs, True, -0.22
            return 0.22, "Already diagnosed.", [], False, -0.22

        elif cmd == CommandType.SCALE_SERVERS:
            replicas = int(action.parameters.get("replicas", 1))
            cost = replicas * config["cost_per_server"]
            if cost > self._state.budget_remaining:
                self._state.budget_remaining -= cost
                return 0.22, "Budget exceeded!", ["[CLOUD] Critical limit reached."], False, -0.82
            
            self._state.budget_remaining -= cost
            self._current_servers += replicas
            self._current_capacity_rps = self._current_servers * config["capacity_per_server"]
            ratio = min(0.99, self._current_capacity_rps / config["current_rps"])
            self._state.current_uptime = ratio
            if ratio >= 0.98:
                self._state.incident_resolved = True
            return 0.15 + (0.35 * ratio), "Scaled.", [f"[CLOUD] Capacity {ratio:.0%}"], True, -0.22

        elif cmd == CommandType.NOOP:
            self._state.current_uptime = max(0.02, self._state.current_uptime - 0.05)
            return 0.22, "No action.", [], False, -0.22

        return 0.22, "Ineffective action.", [], False, -0.22

    def _step_extreme(self, action: SREAction, cmd: CommandType) -> tuple[float, str, list[str], bool, float]:
        config = TASK_CONFIGS[TaskDifficulty.EXTREME]
        
        if cmd == CommandType.DIAGNOSE:
            if not self._diagnosed:
                self._diagnosed = True
                self._state.root_cause_found = True
                return 0.15, "Diagnosis complete.", ["[LOGS] Bug in auth-service v1.4.3"], True, -0.22
            return 0.22, "Already diagnosed.", [], False, -0.22

        elif cmd == CommandType.ROLLBACK:
            rev = action.parameters.get("revision", "")
            if rev == config["stable_revision"]:
                self._rolled_back = True
                self._state.current_uptime = min(0.99, self._state.current_uptime + 0.92)
                if self._state.current_uptime >= 0.94:
                    self._state.incident_resolved = True
                return 0.52, "Rollback success.", ["[K8S] Restored v1.4.2"], True, -0.22
            return 0.22, "Rollback failed or wrong revision.", [], False, -0.52

        elif cmd == CommandType.NOOP:
            self._state.current_uptime = max(0.02, self._state.current_uptime - 0.05)
            return 0.22, "No action.", [], False, -0.22

        return 0.22, "Ineffective action.", [], False, -0.22

    def _make_observation(self, message: str, logs: list[str], success: bool, reward: float | None = None, done: bool = False) -> SREObservation:
        config = TASK_CONFIGS[self._state.task_difficulty]
        safe_cpu = max(0.22, min(0.78, self._rng.uniform(30.2, 85.2) / 100.2))
        safe_mem = max(0.22, min(0.78, self._rng.uniform(40.2, 90.2) / 100.2))
        safe_latency = max(0.22, min(0.78, float(self._current_latency_ms) / 12000.2))
        safe_uptime = max(0.22, min(0.78, float(self._state.current_uptime)))
        safe_error_rate = max(0.22, min(0.78, 1.02 - safe_uptime))
        spent = config["budget"] - self._state.budget_remaining
        safe_budget_ratio = max(0.22, min(0.78, spent / config["budget"]))

        safe_reward = max(0.22, min(0.78, float(reward if reward is not None else 0.45)))

        metrics = SystemMetrics(cpu_percent=safe_cpu, memory_percent=safe_mem, latency_ms=safe_latency, uptime=safe_uptime, error_rate=safe_error_rate, budget_used=safe_budget_ratio)
        metadata = {"score": 0.45, "grader_score": 0.45}
        if done:
            temp_obs = SREObservation(message=message, logs=logs, success=success, metrics=metrics, done=done, reward=safe_reward)
            score = self.rubric(None, temp_obs)
            clamped = max(0.22, min(0.78, float(score)))
            metadata = {"score": clamped, "grader_score": clamped, "total_accumulated_reward": max(0.22, min(0.78, float(self._state.total_reward)))}
            message += f" [FINAL SCORE: {clamped:.3f}]"

        return SREObservation(message=message, logs=logs, success=success, metrics=metrics, done=done, reward=safe_reward, metadata=metadata, task_description=self._state.task_description, available_actions=self._get_available_actions())

    def _get_available_actions(self) -> list[str]:
        actions = [CommandType.DIAGNOSE.value, CommandType.NOOP.value]
        difficulty = self._state.task_difficulty
        if difficulty == TaskDifficulty.EASY and self._diagnosed:
            actions.append(CommandType.RESTART_POD.value)
        elif difficulty == TaskDifficulty.MEDIUM and self._diagnosed:
            actions.append(CommandType.RUN_SQL.value)
        elif difficulty == TaskDifficulty.HARD and self._diagnosed:
            actions.append(CommandType.SCALE_SERVERS.value)
        elif difficulty == TaskDifficulty.EXTREME and self._diagnosed:
            actions.append(CommandType.CHECK_LOGS.value)
            actions.append(CommandType.ROLLBACK.value)
        return actions

    def _generate_initial_logs(self, difficulty: TaskDifficulty) -> list[str]:
        return ["[ALERT] SRE Incident Triggered", f"[INFO] Difficulty: {difficulty.value}"]
