from openenv.core.env_server.http_server import HTTPEnvServer

from server.environment import SREEnvironment
from sre_env.models import SREAction, SREObservation

env_instance = SREEnvironment()

env_server = HTTPEnvServer(
    env=lambda: env_instance,
    action_cls=SREAction,
    observation_cls=SREObservation,
)
