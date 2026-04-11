from typing import Any

from openenv.core.rubrics.base import Rubric

from sre_env.models import TaskDifficulty


def _clamp(score: float) -> float:
    """Clamp score to strictly (0, 1) — never exactly 0.0 or 1.0."""
    return max(0.01, min(0.99, score))


class SREGraderRubric(Rubric):
    def __init__(self, env: Any = None) -> None:
        super().__init__()
        self._env = env

    def forward(self, action: Any, observation: Any) -> float:
        try:
            # Fallback to observation inference if env is completely missing
            if self._env and hasattr(self._env, "state"):
                difficulty = self._env.state.task_difficulty
            elif hasattr(observation, "task_description"):
                desc = observation.task_description
                if "pod-web-3" in desc:
                    difficulty = TaskDifficulty.EASY
                elif "orders" in desc:
                    difficulty = TaskDifficulty.MEDIUM
                elif "spike" in desc or "traffic" in desc:
                    difficulty = TaskDifficulty.HARD
                elif "auth-service" in desc or "cascading" in desc:
                    difficulty = TaskDifficulty.EXTREME
                else:
                    return 0.01
            else:
                return 0.01

            if difficulty == TaskDifficulty.EASY:
                return self._grade_easy(observation)
            elif difficulty == TaskDifficulty.MEDIUM:
                return self._grade_medium(observation)
            elif difficulty == TaskDifficulty.HARD:
                return self._grade_hard(observation)
            elif difficulty == TaskDifficulty.EXTREME:
                return self._grade_extreme(observation)
            return 0.01
        except Exception:
            # Safely catch stateless OpenEnv evaluator invocations
            # and guarantee boundary limits.
            return 0.01

    def _grade_easy(self, observation: Any) -> float:
        """Proportional to restored uptime (target ≥ 0.95)."""
        metrics = getattr(observation, "metrics", None)
        current_uptime = metrics.uptime if metrics else (self._env.state.current_uptime if self._env else 0.0)
        return _clamp(current_uptime / 0.95)

    def _grade_medium(self, observation: Any) -> float:
        """Proportional to latency improvement."""
        from server.environment import TASK_CONFIGS

        config = TASK_CONFIGS[TaskDifficulty.MEDIUM]
        baseline = config["initial_latency_ms"]
        target = config["target_latency_ms"]

        metrics = getattr(observation, "metrics", None)
        current = metrics.latency_ms if metrics else (self._env._current_latency_ms if self._env else baseline)

        if current <= target:
            return 0.99

        improvement_ratio = (baseline - current) / (baseline - target)
        return _clamp(improvement_ratio)

    def _grade_hard(self, observation: Any) -> float:
        """Composite: 50% uptime + 30% budget + 20% speed bonus."""
        from server.environment import TASK_CONFIGS

        config = TASK_CONFIGS[TaskDifficulty.HARD]
        metrics = getattr(observation, "metrics", None)

        current_uptime = metrics.uptime if metrics else (self._env.state.current_uptime if self._env else 0.0)
        
        # Budget logic
        if self._env:
            budget_remaining = self._env.state.budget_remaining
        elif metrics:
            budget_remaining = config["budget"] - metrics.budget_used
        else:
            budget_remaining = 0.0
            
        # Speed logic (approximate if env is missing)
        if self._env:
            step_ratio = self._env.state.step_count / self._env.state.max_steps
        else:
            step_ratio = 1.0

        # Uptime score
        uptime_score = min(1.0, current_uptime / 0.90)

        # Budget score — proportion of budget remaining
        budget_score = max(0.0, budget_remaining / config["budget"])

        # Speed bonus — fewer steps = higher bonus
        speed_bonus = max(0.0, 1.0 - step_ratio)

        return _clamp(0.5 * uptime_score + 0.3 * budget_score + 0.2 * speed_bonus)

    def _grade_extreme(self, observation: Any) -> float:
        """Composite: 70% uptime + 30% speed bonus (No budget used in rollback)."""
        metrics = getattr(observation, "metrics", None)
        current_uptime = metrics.uptime if metrics else (self._env.state.current_uptime if self._env else 0.0)
        
        if self._env:
            step_ratio = self._env.state.step_count / self._env.state.max_steps
        else:
            step_ratio = 1.0
            
        uptime_score = min(1.0, current_uptime / 0.95)
        speed_bonus = max(0.0, 1.0 - step_ratio)
        return _clamp(0.7 * uptime_score + 0.3 * speed_bonus)
