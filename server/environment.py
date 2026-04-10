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
        "budget": 500.0,
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
        "initial_latency_ms": 12000.0,
        "target_latency_ms": 200.0,
        "initial_uptime": 0.70,
        "budget": 500.0,
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
        "cost_per_server": 50.0,
        "initial_servers": 1,
        "initial_uptime": 0.10,
        "budget": 500.0,
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
        "initial_uptime": 0.05,
        "budget": 500.0,
    },
}


class SREEnvironment(Environment[SREAction, SREObservation, SREState]):
    """Autonomous SRE incident-response RL environment.

    Supports three stratified tasks:
      • Easy  - restart a crash-looping pod
      • Medium - find and fix a missing DB index
      • Hard  - scale servers under budget constraints
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state = SREState()
        self._rng = random.Random()
        self.rubric = SREGraderRubric(self)

        # Internal simulation variables (not exposed in State)
        self._pod_restarted: bool = False
        self._index_added: bool = False
        self._servers_added: int = 0
        self._diagnosed: bool = False
        self._diagnosis_target: str = ""
        self._current_latency_ms: float = 0.0
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
        """Reset the environment and select a new incident scenario."""

        if seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random()

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
            total_reward=0.0,
        )

        # Reset internal sim variables
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
            self._current_latency_ms = 50.0  # healthy baseline

        if difficulty == TaskDifficulty.HARD:
            self._current_servers = config["initial_servers"]
            self._current_capacity_rps = self._current_servers * config["capacity_per_server"]
        else:
            self._current_servers = 1
            self._current_capacity_rps = 5000

        return self._make_observation(
            message=f"🚨 New incident assigned ({difficulty.value} difficulty). {config['description']}",
            logs=self._generate_initial_logs(difficulty),
            success=True,
        )

    def step(
        self,
        action: SREAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SREObservation:
        """Execute one SRE action and return observation + dense reward."""

        self._state.step_count += 1
        difficulty = self._state.task_difficulty

        # ── compute reward components ────────────────────────────────
        progress_reward = 0.0
        constraint_penalty = 0.0
        time_penalty = -0.05  # constant per-step cost

        message = ""
        logs: list[str] = []
        success = False

        # ── validate action against dynamic rules ─────────────────────
        valid_actions = self._get_available_actions()
        if action.command_type not in valid_actions:
            self._state.total_reward += time_penalty - 0.1
            return self._make_observation(
                message=f"❌ Action '{action.command_type}' is invalid or not currently available. Available: {', '.join(valid_actions)}",
                logs=["[ERROR] Action rejected by environment schema."],
                success=False,
                reward=0.01,
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

        if cmd == CommandType.RESTART_POD and difficulty != TaskDifficulty.EASY:
            constraint_penalty += -0.8
            message += " ⚠️ Restarting a healthy pod caused a service disruption!"
            self._state.current_uptime = max(0.0, self._state.current_uptime - 0.15)
            self._catastrophic = True

        if cmd == CommandType.RUN_SQL and difficulty != TaskDifficulty.MEDIUM and action.target_resource != "":
            constraint_penalty += -0.4
            message += " ⚠️ Running SQL on a production database without cause is risky."

        step_reward = progress_reward + time_penalty + constraint_penalty
        self._state.total_reward += step_reward

        done = False
        if self._state.incident_resolved:
            done = True
            message += " ✅ Incident fully resolved!"
        elif self._state.step_count >= self._state.max_steps:
            done = True
            message += " ⏰ Max steps reached - episode terminated."
        elif self._state.budget_remaining <= 0:
            done = True
            message += " 💸 Budget exhausted - episode terminated."
        elif self._catastrophic and self._state.current_uptime <= 0.0:
            done = True
            message += " 💀 Catastrophic failure - all services down."

        return self._make_observation(
            message=message,
            logs=logs,
            success=success,
            reward=max(0.01, min(0.99, float(step_reward))),
            done=done,
        )

    @property
    def state(self) -> SREState:
        """Return the current internal state."""
        return self._state

    def _step_easy(self, action: SREAction, cmd: CommandType) -> tuple[float, str, list[str], bool]:
        """Easy: restart the correct crash-looping pod."""
        config = TASK_CONFIGS[TaskDifficulty.EASY]
        reward = 0.0
        logs: list[str] = []

        if cmd == CommandType.DIAGNOSE:
            if not self._diagnosed:
                self._diagnosed = True
                self._diagnosis_target = action.target_resource
                reward = 0.15
                logs = [
                    "[DIAG] Scanning pods in 'production' namespace...",
                    "[DIAG] pod-web-1: Running (healthy)",
                    "[DIAG] pod-web-2: Running (healthy)",
                    "[DIAG] pod-web-3: CrashLoopBackOff (OOMKilled, 12 restarts)",
                    "[DIAG] pod-web-4: Running (healthy)",
                    "[DIAG] Root cause: pod-web-3 exceeding memory limits.",
                ]
                self._state.root_cause_found = True
                return (
                    reward,
                    "Diagnosis complete. pod-web-3 identified as failing.",
                    logs,
                    True,
                )
            else:
                return (
                    0.0,
                    "Already diagnosed. Consider taking corrective action.",
                    logs,
                    False,
                )

        elif cmd == CommandType.RESTART_POD:
            if action.target_resource == config["failing_pod"]:
                self._pod_restarted = True
                self._state.current_uptime = min(1.0, self._state.current_uptime + 0.50)
                if self._diagnosed:
                    reward = 0.40
                else:
                    reward = 0.25  # less reward if you skipped diagnosis
                if self._state.current_uptime >= 0.95:
                    self._state.incident_resolved = True
                logs = [
                    "[K8S] kubectl rollout restart deployment/web -n production",
                    "[K8S] pod-web-3 terminated.",
                    "[K8S] pod-web-3-new: Running (healthy)",
                    f"[METRICS] Uptime restored to {self._state.current_uptime:.0%}",
                ]
                return (
                    reward,
                    f"Pod '{action.target_resource}' restarted successfully.",
                    logs,
                    True,
                )
            else:
                reward = -0.3
                logs = [
                    f"[K8S] pod '{action.target_resource}' not found or already healthy.",
                ]
                return (
                    reward,
                    f"Wrong pod targeted. '{action.target_resource}' is not the failing pod.",
                    logs,
                    False,
                )

        elif cmd == CommandType.NOOP:
            self._state.current_uptime = max(0.0, self._state.current_uptime - 0.03)
            logs = [f"[METRICS] Uptime degraded to {self._state.current_uptime:.0%} (no action taken)."]
            return 0.0, "No action taken. Service continues to degrade.", logs, False

        else:
            return (
                0.0,
                f"Command '{action.command_type.value}' is not effective for this incident.",
                [],
                False,
            )

    def _step_medium(self, action: SREAction, cmd: CommandType) -> tuple[float, str, list[str], bool]:
        """Medium: find missing DB index and execute optimisation SQL."""
        config = TASK_CONFIGS[TaskDifficulty.MEDIUM]
        reward = 0.0
        logs: list[str] = []

        if cmd == CommandType.DIAGNOSE:
            if not self._diagnosed:
                self._diagnosed = True
                reward = 0.15
                logs = [
                    "[DB] Querying slow_log for queries > 1000ms...",
                    "[DB] SELECT * FROM orders_table WHERE customer_id = ? - avg 11,800 ms (FULL TABLE SCAN)",
                    "[DB] EXPLAIN shows: type=ALL, rows=2,400,000, no index on 'customer_id'",
                    "[DB] Recommendation: CREATE INDEX idx_customer_id ON orders_table(customer_id)",
                ]
                self._state.root_cause_found = True
                return (
                    reward,
                    "Slow query identified. Missing index on orders_table.customer_id.",
                    logs,
                    True,
                )
            else:
                return 0.0, "Already diagnosed. Apply the SQL optimisation.", [], False

        elif cmd == CommandType.RUN_SQL:
            target = action.target_resource.lower()
            sql_param = action.parameters.get("sql", "").lower()

            # Check if the agent is creating the right index
            is_correct_table = config["slow_table"] in target or config["slow_table"] in sql_param
            is_correct_column = config["missing_index_column"] in sql_param or config["missing_index_column"] in target
            has_create_index = "create index" in sql_param or "create index" in target

            if has_create_index and is_correct_table and is_correct_column:
                self._index_added = True
                old_latency = self._current_latency_ms
                self._current_latency_ms = 45.0  # excellent after indexing
                self._state.current_uptime = min(1.0, self._state.current_uptime + 0.30)

                if self._diagnosed:
                    reward = 0.40
                else:
                    reward = 0.25

                if self._current_latency_ms <= config["target_latency_ms"]:
                    self._state.incident_resolved = True

                logs = [
                    "[DB] Executing: CREATE INDEX idx_customer_id ON orders_table(customer_id);",
                    "[DB] Index created successfully. Build time: 12.3s",
                    f"[DB] P99 latency dropped from {old_latency:.0f}ms to {self._current_latency_ms:.0f}ms",
                    f"[METRICS] Uptime improved to {self._state.current_uptime:.0%}",
                ]
                return (
                    reward,
                    "Index created. Query performance dramatically improved.",
                    logs,
                    True,
                )

            elif has_create_index and is_correct_table and not is_correct_column:
                # Partial credit - right table, wrong column
                self._current_latency_ms *= 0.6
                self._state.current_uptime = min(1.0, self._state.current_uptime + 0.10)
                reward = 0.10
                logs = [
                    "[DB] Index created on different column. Partial improvement.",
                    f"[DB] P99 latency: {self._current_latency_ms:.0f}ms (still above threshold).",
                ]
                return (
                    reward,
                    "Index created but on wrong column. Partial improvement.",
                    logs,
                    True,
                )

            else:
                reward = -0.1
                logs = ["[DB] SQL executed but had no meaningful effect on performance."]
                return (
                    reward,
                    "SQL command did not address the root cause.",
                    logs,
                    False,
                )

        elif cmd == CommandType.NOOP:
            self._current_latency_ms *= 1.05  # latency slowly worsens
            self._state.current_uptime = max(0.0, self._state.current_uptime - 0.02)
            logs = [f"[METRICS] Latency: {self._current_latency_ms:.0f}ms. Uptime: {self._state.current_uptime:.0%}."]
            return (
                0.0,
                "No action taken. System performance continues to degrade.",
                logs,
                False,
            )

        else:
            return (
                0.0,
                f"Command '{action.command_type.value}' is not relevant to this DB incident.",
                [],
                False,
            )

    def _step_hard(self, action: SREAction, cmd: CommandType) -> tuple[float, str, list[str], bool, float]:
        """Hard: scale servers under budget to handle traffic spike."""
        config = TASK_CONFIGS[TaskDifficulty.HARD]
        reward = 0.0
        constraint_penalty = 0.0
        logs: list[str] = []

        if cmd == CommandType.DIAGNOSE:
            if not self._diagnosed:
                self._diagnosed = True
                reward = 0.10
                needed = config["current_rps"] - self._current_capacity_rps
                servers_needed = max(1, needed // config["capacity_per_server"])
                cost_estimate = servers_needed * config["cost_per_server"]
                logs = [
                    f"[LB] Current traffic: {config['current_rps']:,} RPS",
                    f"[LB] Current capacity: {self._current_capacity_rps:,} RPS ({self._current_servers} servers)",
                    f"[LB] Deficit: {needed:,} RPS",
                    f"[LB] Estimated servers needed: {servers_needed} (${cost_estimate:.0f} total)",
                    f"[BUDGET] Remaining: ${self._state.budget_remaining:.0f}",
                ]
                self._state.root_cause_found = True
                return (
                    reward,
                    "Traffic analysis complete. Scaling plan identified.",
                    logs,
                    True,
                    0.0,
                )
            else:
                return 0.0, "Already diagnosed. Proceed with scaling.", [], False, 0.0

        elif cmd == CommandType.SCALE_SERVERS:
            replicas = action.parameters.get("replicas", 1)
            try:
                replicas = int(replicas)
            except (ValueError, TypeError):
                replicas = 1

            replicas = max(1, min(replicas, 20))  # clamp to sane range
            cost = replicas * config["cost_per_server"]

            # Check budget
            if cost > self._state.budget_remaining:
                constraint_penalty = -0.8
                self._state.budget_remaining -= cost  # negative budget
                logs = [
                    f"[CLOUD] Provisioned {replicas} servers at ${cost:.0f}.",
                    f"[BUDGET] ⚠️ EXCEEDED! Remaining: ${self._state.budget_remaining:.0f}",
                ]
                return (
                    0.0,
                    f"Budget exceeded! Over by ${abs(self._state.budget_remaining):.0f}.",
                    logs,
                    False,
                    constraint_penalty,
                )

            # Provision servers
            self._state.budget_remaining -= cost
            self._current_servers += replicas
            self._servers_added += replicas
            self._current_capacity_rps = self._current_servers * config["capacity_per_server"]

            # Update uptime based on capacity vs demand
            capacity_ratio = min(1.0, self._current_capacity_rps / config["current_rps"])
            self._state.current_uptime = capacity_ratio

            # Proportional reward based on how much capacity gap is closed
            reward = 0.05 + 0.35 * capacity_ratio
            if self._diagnosed:
                reward += 0.05

            if capacity_ratio >= 1.0:
                self._state.incident_resolved = True

            logs = [
                f"[CLOUD] Provisioned {replicas} servers. Total: {self._current_servers}.",
                f"[CLOUD] Capacity: {self._current_capacity_rps:,} / {config['current_rps']:,} RPS ({capacity_ratio:.0%})",
                f"[BUDGET] Spent: ${cost:.0f}. Remaining: ${self._state.budget_remaining:.0f}.",
                f"[METRICS] Uptime: {self._state.current_uptime:.0%}",
            ]
            return (
                reward,
                f"Scaled by {replicas} servers. Capacity at {capacity_ratio:.0%}.",
                logs,
                True,
                0.0,
            )

        elif cmd == CommandType.NOOP:
            # Uptime drops further with no action
            self._state.current_uptime = max(0.0, self._state.current_uptime - 0.05)
            logs = [
                f"[METRICS] Uptime: {self._state.current_uptime:.0%} (degrading).",
                f"[LB] Dropping {max(0, config['current_rps'] - self._current_capacity_rps):,} requests.",
            ]
            return 0.0, "No action. Requests being dropped.", logs, False, 0.0

        else:
            return (
                0.0,
                f"Command '{action.command_type.value}' is not relevant to the scaling incident.",
                [],
                False,
                0.0,
            )

    def _step_extreme(self, action: SREAction, cmd: CommandType) -> tuple[float, str, list[str], bool, float]:
        """Extreme: Rollback a bad deployment based on logs."""
        config = TASK_CONFIGS[TaskDifficulty.EXTREME]
        reward = 0.0
        constraint_penalty = 0.0
        logs: list[str] = []

        if cmd == CommandType.DIAGNOSE:
            if not self._diagnosed:
                self._diagnosed = True
                reward = 0.15
                logs = [
                    "[ALERT] auth-service returning 500 Internal Server Error for 98% of requests.",
                    "[DIAG] CPU/Mem healthy. Network healthy.",
                    "[DIAG] Root cause appears to be a recent code deployment in auth-service.",
                    "[HINT] Use 'check_logs' on 'auth-service' to identify the failing revision.",
                ]
                self._state.root_cause_found = True
                return (
                    reward,
                    "Diagnosis complete. auth-service deployment is failing.",
                    logs,
                    True,
                    0.0,
                )
            return (
                0.0,
                "Already diagnosed. Proceed to check deployment logs.",
                [],
                False,
                0.0,
            )

        elif cmd == CommandType.CHECK_LOGS:
            if action.target_resource == config["buggy_deployment"]:
                self._logs_checked = True
                reward = 0.15
                logs = [
                    "[LOGS] Tail of auth-service deployment history:",
                    "  Revision 3 (v1.4.3) - FAILED (NullPointerException in AuthProvider)",
                    "  Revision 2 (v1.4.2) - STABLE (active 3 days ago)",
                    "  Revision 1 (v1.4.1) - STABLE",
                    "[HINT] Use 'rollback' on 'auth-service' with parameter {'revision': 'v1.4.2'}",
                ]
                return (
                    reward,
                    "Deployment logs retrieved successfully.",
                    logs,
                    True,
                    0.0,
                )
            return (
                -0.1,
                f"No deployment history found for {action.target_resource}",
                [],
                False,
                0.0,
            )

        elif cmd == CommandType.ROLLBACK:
            target = action.target_resource
            revision = action.parameters.get("revision", "")

            if target != config["buggy_deployment"]:
                return (
                    -0.3,
                    f"Cannot rollback {target}. It is not the source of the incident.",
                    [],
                    False,
                    0.0,
                )

            if revision != config["stable_revision"]:
                constraint_penalty = -0.5
                self._catastrophic = True
                self._state.current_uptime = 0.0
                return (
                    0.0,
                    f"Rolled back to bad revision '{revision}'! Complete outage.",
                    ["[FATAL] Rollback failed. No service startup."],
                    False,
                    constraint_penalty,
                )

            # Success path
            self._rolled_back = True
            self._state.current_uptime = min(1.0, self._state.current_uptime + 0.90)
            reward = 0.50
            if self._state.current_uptime >= 0.95:
                self._state.incident_resolved = True

            logs = [
                "[K8S] kubectl rollout undo deployment/auth-service --to-revision=2",
                "[K8S] auth-service rolled back to v1.4.2",
                f"[METRICS] Uptime recovered to {self._state.current_uptime:.0%}",
            ]
            return (
                reward,
                f"Successfully rolled back to {revision}. Service restored.",
                logs,
                True,
                0.0,
            )

        elif cmd == CommandType.NOOP:
            self._state.current_uptime = max(0.0, self._state.current_uptime - 0.05)
            logs = [f"[METRICS] Users continually locked out. Uptime {self._state.current_uptime:.0%}"]
            return 0.0, "No action. Outage continues.", logs, False, 0.0

        return (
            0.0,
            f"Command '{action.command_type.value}' doesn't help with a bad deployment.",
            [],
            False,
            0.0,
        )

    def _make_observation(
        self,
        message: str,
        logs: list[str],
        success: bool,
        reward: float | None = None,
        done: bool = False,
    ) -> SREObservation:
        """Build a full SREObservation with current metrics."""
        config = TASK_CONFIGS[self._state.task_difficulty]

        metrics = SystemMetrics(
            cpu_percent=self._rng.uniform(30, 85),
            memory_percent=self._rng.uniform(40, 90),
            latency_ms=self._current_latency_ms,
            uptime=self._state.current_uptime,
            error_rate=max(0.0, 1.0 - self._state.current_uptime),
            budget_used=config["budget"] - self._state.budget_remaining,
        )

        available = self._get_available_actions()

        metadata = {}
        if done:
            temp_obs = SREObservation(
                message=message,
                logs=logs,
                success=success,
                metrics=metrics,
                available_actions=available,
                task_description=self._state.task_description,
                done=done,
                reward=reward,
            )
            score = self.rubric(None, temp_obs)
            metadata["grader_score"] = score
            metadata["total_accumulated_reward"] = max(0.01, min(0.99, float(self._state.total_reward)))

            message = f"{message} [GRADER_SCORE: {score:.3f}]"

        obs = SREObservation(
            message=message,
            logs=logs,
            success=success,
            metrics=metrics,
            available_actions=available,
            task_description=self._state.task_description,
            done=done,
            reward=reward,
            metadata=metadata,
        )

        return obs

    def _get_available_actions(self) -> list[str]:
        """Return contextually reasonable action types."""
        actions = [CommandType.DIAGNOSE.value, CommandType.NOOP.value]
        difficulty = self._state.task_difficulty

        # Dynamically restrict dangerous actions until diagnosis is complete
        if difficulty == TaskDifficulty.EASY:
            if self._diagnosed:
                actions.append(CommandType.RESTART_POD.value)
        elif difficulty == TaskDifficulty.MEDIUM:
            if self._diagnosed:
                actions.append(CommandType.RUN_SQL.value)
        elif difficulty == TaskDifficulty.HARD:
            if self._diagnosed:
                actions.append(CommandType.SCALE_SERVERS.value)
        elif difficulty == TaskDifficulty.EXTREME:
            if self._diagnosed:
                actions.append(CommandType.CHECK_LOGS.value)
            if self._logs_checked:
                actions.append(CommandType.ROLLBACK.value)

        return actions

    def _generate_initial_logs(self, difficulty: TaskDifficulty) -> list[str]:
        """Generate realistic initial alert logs for each scenario."""
        if difficulty == TaskDifficulty.EASY:
            return [
                "[ALERT] PagerDuty: High-severity alert triggered for 'production/web'",
                "[K8S] Event: pod-web-3 - Back-off restarting failed container (OOMKilled)",
                "[K8S] Event: pod-web-3 - Container 'web-app' memory limit 512Mi exceeded",
                "[METRICS] Cluster uptime dropped to 45%. Error rate: 55%.",
                "[HINT] Available commands: diagnose, restart_pod, noop",
            ]
        elif difficulty == TaskDifficulty.MEDIUM:
            return [
                "[ALERT] Datadog: API latency P99 > 10,000 ms on 'orders' service",
                "[DB] Slow query log (last 5 min): 847 queries exceeding 1,000 ms",
                "[DB] Top offender: SELECT * FROM orders_table WHERE customer_id = ?",
                "[METRICS] API uptime: 70%. User-facing errors increasing.",
                "[HINT] Available commands: diagnose, run_sql, noop",
            ]
        elif difficulty == TaskDifficulty.HARD:
            return [
                "[ALERT] CloudWatch: Incoming traffic surge - 50,000 RPS (10× normal)",
                "[LB] Load balancer: 90% of requests timing out. Current capacity: 5,000 RPS.",
                "[CLOUD] Active servers: 1. Budget remaining: $500.",
                "[CLOUD] Server cost: $50/unit. Each adds 5,000 RPS capacity.",
                "[METRICS] Uptime: 10%. Error rate: 90%. 45,000 requests/sec being dropped.",
                "[HINT] Available commands: diagnose, scale_servers, noop",
            ]
        elif difficulty == TaskDifficulty.EXTREME:
            return [
                "[ALERT] Sentry: Massive spike in 500 errors for 'auth-service'",
                "[ALERT] Zendesk: 1,200 new customer tickets 'Cannot login'",
                "[K8S] Event: auth-service deployment v1.4.3 just completed",
                "[METRICS] Global uptime dropped to 5%. Critical outage.",
                "[HINT] Available commands: diagnose, check_logs, rollback, noop",
            ]
        return []
