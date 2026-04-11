from typing import Any

from openenv.core.rubrics.base import Rubric

from sre_env.models import TaskDifficulty


def _clamp(score: float) -> float:
    """Clamp score to strictly (0, 1) — never exactly 0.0 or 1.0. Hardened to prevent rounded 1.0s."""
    return max(0.05, min(0.95, float(score)))

def _extract_metric(observation: Any, key: str, default: float) -> float:
    """Safely extract a metric from either a Pydantic object or a generic nested dictionary."""
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
            fallback = self._env.state.current_uptime if self._env else 0.0
            current_uptime = _extract_metric(observation, "uptime", fallback)
            return _clamp(current_uptime / 0.95)
        except Exception:
            return 0.05

class DBIndexRubric(Rubric):
    def __init__(self, env: Any = None) -> None:
        super().__init__()
        self._env = env

    def forward(self, action: Any, observation: Any) -> float:
        try:
            from server.environment import TASK_CONFIGS
            config = TASK_CONFIGS[TaskDifficulty.MEDIUM]
            baseline = config["initial_latency_ms"]
            target = config["target_latency_ms"]

            fallback = self._env._current_latency_ms if self._env else baseline
            current = _extract_metric(observation, "latency_ms", fallback)

            if current <= target:
                return 0.95

            improvement_ratio = (baseline - current) / (baseline - target)
            return _clamp(improvement_ratio)
        except Exception:
            return 0.05

class ScalingRubric(Rubric):
    def __init__(self, env: Any = None) -> None:
        super().__init__()
        self._env = env

    def forward(self, action: Any, observation: Any) -> float:
        try:
            from server.environment import TASK_CONFIGS
            config = TASK_CONFIGS[TaskDifficulty.HARD]

            uptime_fallback = self._env.state.current_uptime if self._env else 0.0
            current_uptime = _extract_metric(observation, "uptime", uptime_fallback)
            
            if self._env:
                budget_remaining = self._env.state.budget_remaining
            else:
                raw_used = _extract_metric(observation, "budget_used", 0.0)
                budget_remaining = config["budget"] - raw_used
                
            if self._env:
                step_ratio = self._env.state.step_count / self._env.state.max_steps
            else:
                step_ratio = 1.0

            uptime_score = min(1.0, current_uptime / 0.90)
            budget_score = max(0.0, budget_remaining / config["budget"])
            speed_bonus = max(0.0, 1.0 - step_ratio)

            return _clamp(0.5 * uptime_score + 0.3 * budget_score + 0.2 * speed_bonus)
        except Exception:
            return 0.05

class RollbackRubric(Rubric):
    def __init__(self, env: Any = None) -> None:
        super().__init__()
        self._env = env

    def forward(self, action: Any, observation: Any) -> float:
        try:
            uptime_fallback = self._env.state.current_uptime if self._env else 0.0
            current_uptime = _extract_metric(observation, "uptime", uptime_fallback)
            
            if self._env:
                step_ratio = self._env.state.step_count / self._env.state.max_steps
            else:
                step_ratio = 1.0
                
            uptime_score = min(1.0, current_uptime / 0.95)
            speed_bonus = max(0.0, 1.0 - step_ratio)
            return _clamp(0.7 * uptime_score + 0.3 * speed_bonus)
        except Exception:
            return 0.05

class SREGraderRubric(Rubric):
    """Facade for the Environment to dynamically grade based on actual current state."""
    def __init__(self, env: Any = None) -> None:
        super().__init__()
        self._env = env

    def forward(self, action: Any, observation: Any) -> float:
        try:
            if not self._env:
                return 0.05
            
            diff = getattr(self._env, "state", None)
            if diff:
                diff = diff.task_difficulty
            
            if diff == TaskDifficulty.EASY:
                return PodRestartRubric(self._env).forward(action, observation)
            elif diff == TaskDifficulty.MEDIUM:
                return DBIndexRubric(self._env).forward(action, observation)
            elif diff == TaskDifficulty.HARD:
                return ScalingRubric(self._env).forward(action, observation)
            elif diff == TaskDifficulty.EXTREME:
                return RollbackRubric(self._env).forward(action, observation)
            return 0.05
        except Exception:
            return 0.05
