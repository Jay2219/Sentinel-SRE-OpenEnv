from server.environment import SREEnvironment
from sre_env.models import SREAction


def test_easy_episode():
    env = SREEnvironment()
    env.reset(seed=42)
    # Ensure it's easy by forcing it if needed, or by injecting state
    env.state.task_difficulty = "easy"

    # 1. Diagnose
    obs = env.step(SREAction(command_type="diagnose"))
    assert obs.success is True
    assert "pod-web-3" in obs.message

    # 2. Restart wrong pod
    obs2 = env.step(SREAction(command_type="restart_pod", target_resource="pod-web-not-real"))
    assert obs2.success is False
    assert obs2.reward < 1.0

    # 3. Restart correct pod
    obs3 = env.step(SREAction(command_type="restart_pod", target_resource="pod-web-3"))
    assert obs3.success is True
    assert obs3.done is True
    assert obs3.metadata["grader_score"] > 0.8


def test_medium_episode():
    env = SREEnvironment()
    env.reset(seed=42)
    env.state.task_difficulty = "medium"

    # 1. Diagnose
    env.step(SREAction(command_type="diagnose"))

    # 2. Run wrong SQL
    obs = env.step(
        SREAction(
            command_type="run_sql",
            target_resource="users",
            parameters={"sql": "CREATE INDEX on users"},
        )
    )
    assert obs.success is False

    # 3. Run partial SQL (wrong column)
    obs2 = env.step(
        SREAction(
            command_type="run_sql",
            target_resource="orders_table",
            parameters={"sql": "CREATE INDEX on orders_table(date)"},
        )
    )
    assert obs2.success is True
    assert obs2.done is False

    # 4. Run correct SQL
    obs3 = env.step(
        SREAction(
            command_type="run_sql",
            target_resource="orders_table",
            parameters={"sql": "CREATE INDEX idx ON orders_table(customer_id)"},
        )
    )
    assert obs3.success is True
    assert obs3.done is True
    assert obs3.metadata["grader_score"] > 0.9


def test_hard_episode():
    env = SREEnvironment()
    env.reset(seed=42)
    env.state.task_difficulty = "hard"

    env.step(SREAction(command_type="diagnose"))

    # Scale too many (break budget)
    obs = env.step(
        SREAction(
            command_type="scale_servers",
            target_resource="cluster",
            parameters={"replicas": 20},
        )
    )
    assert obs.success is False
    assert obs.done is True  # Out of budget
    assert obs.metadata["grader_score"] < 0.5


def test_extreme_episode():
    env = SREEnvironment()
    env.reset(seed=42)
    env.state.task_difficulty = "extreme"

    # 1. Diagnose
    env.step(SREAction(command_type="diagnose"))

    # 2. Check Logs
    obs_logs = env.step(SREAction(command_type="check_logs", target_resource="auth-service"))
    assert obs_logs.success is True
    assert "v1.4.2" in str(obs_logs.logs)

    # 3. Rollback
    obs_rollback = env.step(
        SREAction(
            command_type="rollback",
            target_resource="auth-service",
            parameters={"revision": "v1.4.2"},
        )
    )
    assert obs_rollback.success is True
    assert obs_rollback.done is True
    assert obs_rollback.metadata["grader_score"] > 0.8
