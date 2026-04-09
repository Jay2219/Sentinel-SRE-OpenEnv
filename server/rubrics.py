from typing import Any

from openenv.core.rubrics.base import Rubric

from sre_env.models import TaskDifficulty


def _clamp(score: float) -> float:
    """Clamp score to strictly (0, 1) — never exactly 0.0 or 1.0."""
    return max(0.01, min(0.99, score))


class SREGraderRubric(Rubric):
    def __init__(self, env: Any) -> None:
        super().__init__()
        self._env = env

    def forward(self, action: Any, observation: Any) -> float:
        difficulty = self._env.state.task_difficulty

        if difficulty == TaskDifficulty.EASY:
            return self._grade_easy()
        elif difficulty == TaskDifficulty.MEDIUM:
            return self._grade_medium()
        elif difficulty == TaskDifficulty.HARD:
            return self._grade_hard()
        elif difficulty == TaskDifficulty.EXTREME:
            return self._grade_extreme()
        return 0.01

    def _grade_easy(self) -> float:
        """Proportional to restored uptime (target ≥ 0.95)."""
        return _clamp(self._env.state.current_uptime / 0.95)

    def _grade_medium(self) -> float:
        """Proportional to latency improvement."""
        from server.environment import TASK_CONFIGS

        config = TASK_CONFIGS[TaskDifficulty.MEDIUM]
        baseline = config["initial_latency_ms"]
        target = config["target_latency_ms"]

        # Access internal environment state
        current = self._env._current_latency_ms

        if current <= target:
            return 0.99

        improvement_ratio = (baseline - current) / (baseline - target)
        return _clamp(improvement_ratio)

    def _grade_hard(self) -> float:
        """Composite: 50% uptime + 30% budget + 20% speed bonus."""
        from server.environment import TASK_CONFIGS

        config = TASK_CONFIGS[TaskDifficulty.HARD]

        # Uptime score
        uptime_score = min(1.0, self._env.state.current_uptime / 0.90)

        # Budget score — proportion of budget remaining
        budget_score = max(0.0, self._env.state.budget_remaining / config["budget"])

        # Speed bonus — fewer steps = higher bonus
        speed_bonus = max(0.0, 1.0 - (self._env.state.step_count / self._env.state.max_steps))

        return _clamp(0.5 * uptime_score + 0.3 * budget_score + 0.2 * speed_bonus)

    def _grade_extreme(self) -> float:
        """Composite: 70% uptime + 30% speed bonus (No budget used in rollback)."""
        uptime_score = min(1.0, self._env.state.current_uptime / 0.95)
        speed_bonus = max(0.0, 1.0 - (self._env.state.step_count / self._env.state.max_steps))
        return _clamp(0.7 * uptime_score + 0.3 * speed_bonus)
