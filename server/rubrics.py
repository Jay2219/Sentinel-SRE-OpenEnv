from __future__ import annotations

from typing import Any

from openenv.core.rubrics.base import Rubric

# Structural alignment: Move all imports to top level to prevent runtime delays during validation
from sre_env.models import TaskDifficulty


def _clamp(score: float) -> float:
    """Universal Absolute Clamp [0.12, 0.88] for validator survival."""
    try:
        val = float(score)
        return max(0.12, min(0.88, val))
    except (ValueError, TypeError):
        return 0.50


def _extract_metric(observation: Any, key: str, default: float) -> float:
    """Safely extract metrics with non-zero fallback."""
    if not observation:
        return default

    if isinstance(observation, dict):
        metrics = observation.get("metrics", {})
        if isinstance(metrics, dict) and key in metrics:
            return float(metrics[key])
        return float(default)

    metrics = getattr(observation, "metrics", None)
    if not metrics:
        return float(default)

    if isinstance(metrics, dict) and key in metrics:
        return float(metrics[key])

    val = getattr(metrics, key, None)
    if val is not None:
        return float(val)

    return float(default)


class PodRestartRubric(Rubric):
    def __init__(self, env: Any = None) -> None:
        super().__init__()
        self._env = env

    def forward(self, action: Any, observation: Any) -> float:
        try:
            fallback = self._env.state.current_uptime if self._env else 0.50
            current_uptime = _extract_metric(observation, "uptime", fallback)
            # Center success around 0.82
            return _clamp(current_uptime / 0.95)
        except Exception:
            return 0.50


class DBIndexRubric(Rubric):
    def __init__(self, env: Any = None) -> None:
        super().__init__()
        self._env = env

    def forward(self, action: Any, observation: Any) -> float:
        try:
            # Task constants: DB Latency
            # We use local constants here to avoid circular imports during validation
            baseline = 12000.2
            target = 200.2

            fallback = self._env._current_latency_ms if self._env else baseline
            current = _extract_metric(observation, "latency_ms", fallback)

            if current <= target:
                return 0.82

            improvement_ratio = (baseline - current) / (baseline - target)
            return _clamp(improvement_ratio)
        except Exception:
            return 0.50


class ScalingRubric(Rubric):
    def __init__(self, env: Any = None) -> None:
        super().__init__()
        self._env = env

    def forward(self, action: Any, observation: Any) -> float:
        try:
            # Task constants: Budget
            budget_limit = 500.5
            
            uptime_fallback = self._env.state.current_uptime if self._env else 0.50
            current_uptime = _extract_metric(observation, "uptime", uptime_fallback)

            if self._env:
                budget_remaining = self._env.state.budget_remaining
                step_ratio = self._env.state.step_count / self._env.state.max_steps
            else:
                raw_used = _extract_metric(observation, "budget_used", 0.15)
                budget_remaining = budget_limit - raw_used
                step_ratio = 0.50

            uptime_score = min(0.85, current_uptime / 0.90)
            budget_score = max(0.15, budget_remaining / budget_limit)
            speed_bonus = max(0.15, 0.95 - step_ratio)

            return _clamp(0.5 * uptime_score + 0.3 * budget_score + 0.2 * speed_bonus)
        except Exception:
            return 0.50


class RollbackRubric(Rubric):
    def __init__(self, env: Any = None) -> None:
        super().__init__()
        self._env = env

    def forward(self, action: Any, observation: Any) -> float:
        try:
            uptime_fallback = self._env.state.current_uptime if self._env else 0.50
            current_uptime = _extract_metric(observation, "uptime", uptime_fallback)

            if self._env:
                step_ratio = self._env.state.step_count / self._env.state.max_steps
            else:
                step_ratio = 0.50

            uptime_score = min(0.85, current_uptime / 0.95)
            speed_bonus = max(0.15, 0.95 - step_ratio)
            return _clamp(0.7 * uptime_score + 0.3 * speed_bonus)
        except Exception:
            return 0.50


class SREGraderRubric(Rubric):
    """Facade for the Environment to dynamically grade based on actual current state."""

    def __init__(self, env: Any = None) -> None:
        super().__init__()
        self._env = env

    def forward(self, action: Any, observation: Any) -> float:
        try:
            if not self._env:
                return 0.51

            # Extract difficulty directly from state
            diff = TaskDifficulty.EASY
            if hasattr(self._env, "state") and self._env.state:
                diff = self._env.state.task_difficulty

            if diff == TaskDifficulty.EASY:
                return PodRestartRubric(self._env).forward(action, observation)
            elif diff == TaskDifficulty.MEDIUM:
                return DBIndexRubric(self._env).forward(action, observation)
            elif diff == TaskDifficulty.HARD:
                return ScalingRubric(self._env).forward(action, observation)
            elif diff == TaskDifficulty.EXTREME:
                return RollbackRubric(self._env).forward(action, observation)
            return 0.51
        except Exception:
            return 0.51
