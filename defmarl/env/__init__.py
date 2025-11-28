from typing import Optional

from .base import MultiAgentEnv
from .mpe_target import MPETarget
from .mpe_spread import MPESpread
from .mpe_line import MPELine
from .mpe_formation import MPEFormation
from .mpe_corridor import MPECorridor
from .mpe_connect_spread import MPEConnectSpread
from .mpe_inspect import MPEInspect
from .mpe_inspect_slow import MPEInspect_Slow
from .mpe_inspect_bigger_obstacle import MPEInspect_BiggerObstacle
from .mpe_inspect_slow_4agent import MPEInspect_Slow_4Agent
from .mpe_lifelong import MPELifelong

ENV = {
    'MPETarget': MPETarget,
    'MPESpread': MPESpread,
    'MPELine': MPELine,
    'MPEFormation': MPEFormation,
    'MPECorridor': MPECorridor,
    'MPEConnectSpread': MPEConnectSpread,
    'MPEInspect': MPEInspect,
    'MPEInspect_Slow': MPEInspect_Slow,
    'MPEInspect_Big': MPEInspect_BiggerObstacle,
    'MPEInspect_Slow_4Agent': MPEInspect_Slow_4Agent,
    'MPELifelong': MPELifelong
}


DEFAULT_MAX_STEP = 128


def make_env(
        env_id: str,
        num_agents: int,
        area_size: float = None,
        max_step: int = None,
        full_observation: bool = False,
        max_travel: Optional[float] = None,
        num_obs: Optional[int] = None,
) -> MultiAgentEnv:
    assert env_id in ENV.keys(), f'Environment {env_id} not implemented.'
    params = ENV[env_id].PARAMS
    max_step = DEFAULT_MAX_STEP if max_step is None else max_step
    if num_obs is not None:
        params['n_obs'] = num_obs
    if full_observation:
        area_size = params['default_area_size'] if area_size is None else area_size
        params['comm_radius'] = area_size * 10
    return ENV[env_id](
        num_agents=num_agents,
        area_size=area_size,
        max_step=max_step,
        max_travel=max_travel,
        params=params
    )
