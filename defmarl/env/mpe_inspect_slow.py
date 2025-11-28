import pathlib
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import functools as ft

from typing import NamedTuple, Tuple, Optional
from abc import ABC, abstractmethod, abstractproperty

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

from ..trainer.data import Rollout
from ..utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ..utils.typing import Action, Array, Cost, Done, Info, Reward, State, AgentState
from jaxtyping import Float
from ..utils.utils import tree_index, MutablePatchCollection, save_anim
from .base import MultiAgentEnv
from .utils import get_node_goal_rng


class MPEEnvState(NamedTuple):
    agent: State
    goal: State
    obs: State
    time: Float[Array, ""]

    @property
    def n_agent(self) -> int:
        return self.agent.shape[0]


MPEEnvGraphsTuple = GraphsTuple[State, MPEEnvState]


class MPEInspect_Slow(MultiAgentEnv, ABC):

    AGENT = 0
    GOAL = 1
    OBS = 2

    PARAMS = {
        "car_radius": 0.1,
        "comm_radius": 2.0,
        "n_obs": 2,
        "obs_radius": 0.33, ####0.2 -> 0.33
        "default_area_size": 4.0,
        # "dist2goal": 0.01
    }

    def __init__(
            self,
            num_agents: int,
            area_size: Optional[float] = None,
            max_step: int = 1200,   #60 seconds
            max_travel: Optional[float] = None,
            dt: float = 0.05,
            params: dict = None
    ):
        area_size = MPEInspect_Slow.PARAMS["default_area_size"]
        max_step = 1200  # 60 seconds
        dt = 0.05
        super(MPEInspect_Slow, self).__init__(num_agents, area_size, max_step, max_travel, dt, params)
        if self.params["n_obs"] != 2:
            self.params["n_obs"] = 2
            print("WARNING: n_obs is set to 2 for MPEInspect_Slow.")
        self.obstacle_states = jnp.array([
            [1.5, 1.5, 0.0, 0.0],
            [2.5, 2.5, 0.0, 0.0]
        ], dtype=jnp.float32)
        self.obstacle_size = self._params["obs_radius"]
        self.rl_agent_states = jnp.array([
            [0.5, 0.5, 0.0, 0.0],
            [1.5, 0.5, 0.0, 0.0],
            [2.5, 0.5, 0.0, 0.0],
            [3.5, 0.5, 0.0, 0.0],
            [0.5, 1.5, 0.0, 0.0],
            [2.5, 1.5, 0.0, 0.0],
            [3.5, 1.5, 0.0, 0.0],
            [0.5, 2.5, 0.0, 0.0],
            [1.5, 2.5, 0.0, 0.0],
            [3.5, 2.5, 0.0, 0.0],
            [0.5, 3.5, 0.0, 0.0],
            [1.5, 3.5, 0.0, 0.0],
            [2.5, 3.5, 0.0, 0.0],
            [3.5, 3.5, 0.0, 0.0]
        ])
        # self.rl_agent_states = jnp.array([[3.5, 0.5, 0.0, 0.0], [0.5, 3.5, 0.0, 0.0]], dtype=jnp.float32)
        self.non_rl_agent_states = jnp.array([[2.0, 2.0, 0.0, 1.0]], dtype=jnp.float32)

    @property
    def state_dim(self) -> int:
        return 4  # x, y, vx, vy

    @property
    def node_dim(self) -> int:
        return 7  # state dim (4) + indicator: agent: 001, goal: 010, obstacle: 100

    @property
    def edge_dim(self) -> int:
        return 4  # x_rel, y_rel, vx_rel, vy_rel

    @property
    def action_dim(self) -> int:
        return 2  # ax, ay

    @property
    def reward_min(self) -> float:
        return -(0.01 + 0.1 * (self.area_size * np.sqrt(2)) + 0.0001) * self.max_episode_steps

    @property
    def reward_max(self) -> float:
        return 0.5

    @property
    def n_cost(self) -> int:
        return 3
    
    @property
    def non_rl_agents(self) -> int:
        return 1

    @property
    def cost_components(self) -> Tuple[str, ...]:
        return "agent collisions", "target collisions", "obs collisions"

    def reset(self, key: Array) -> GraphsTuple:
        time = jr.uniform(key, (1,), minval=0.0, maxval=6 * jnp.sqrt(2) * jnp.pi)

        def get_rl_agent_states(inp):
            this_key, _ , time= inp
            use_key, this_key = jr.split(this_key, 2)
            return this_key, self.rl_agent_states[jr.choice(use_key, self.rl_agent_states.shape[0], (2,))], time
        def non_valid_rl_agent_states(inp):
            _, this_rl_agent_states, time = inp
            dist_min_agents = jnp.min(jnp.array([jnp.linalg.norm(this_rl_agent_states[0, :2] - this_rl_agent_states[1, :2]),
                                                jnp.linalg.norm(this_rl_agent_states[0, :2] - self.non_rl_agent_trajectory(time)[0, :2]),
                                                jnp.linalg.norm(this_rl_agent_states[1, :2] - self.non_rl_agent_trajectory(time)[0, :2])]))
            return dist_min_agents <= 0.1        
        # Just run once to get 2 non-overlapping agents
        use_key, _ = jr.split(key, 2)
        rl_agent_states_candidate = self.rl_agent_states[jr.choice(use_key, self.rl_agent_states.shape[0], (2,))]
        
        _, rl_agent_states, _ = jax.lax.while_loop(
            non_valid_rl_agent_states, 
            get_rl_agent_states, 
            (key, rl_agent_states_candidate,time)
        )
        
        env_state = MPEEnvState(rl_agent_states, self.non_rl_agent_states, self.obstacle_states, time)
        return self.get_graph(env_state)

    def agent_step_euler(self, agent_states: AgentState, action: Action) -> AgentState:
        # assert action.shape == (self.num_agents, self.action_dim)
        # assert agent_states.shape == (self.num_agents, self.state_dim)
        x_dot = jnp.concatenate([agent_states[:, 2:], action * 10.], axis=1)
        n_state_agent_new = x_dot * self.dt + agent_states
        # assert n_state_agent_new.shape == (self.num_agents, self.state_dim)
        return self.clip_state(n_state_agent_new)

    def non_rl_agent_trajectory(self,t):
        a = 1.5
        w = 1 / jnp.sqrt(2) / a
        rotationangle = jnp.pi / 4
        x = a * jnp.sin(w * t)
        y = a / 2 * jnp.sin(2*w*t)
        vx = a * w * jnp.cos(w * t)
        vy = a * w * jnp.cos(2*w*t)
        rot_x = x * jnp.cos(rotationangle) - y * jnp.sin(rotationangle) + 2.0
        rot_y = x * jnp.sin(rotationangle) + y * jnp.cos(rotationangle) + 2.0
        rot_vx = vx * jnp.cos(rotationangle) - vy * jnp.sin(rotationangle)
        rot_vy = vx * jnp.sin(rotationangle) + vy * jnp.cos(rotationangle)
        return jnp.array([[rot_x, rot_y, rot_vx, rot_vy]], dtype=jnp.float32).squeeze(-1)
        

    def step(
            self, graph: MPEEnvGraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[MPEEnvGraphsTuple, Reward, Cost, Done, Info]:
        # get information from graph
        rl_agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        # non_rl_agent_states = graph.type_states(type_idx=1, n_type=self.non_rl_agents)
        obstacles = graph.type_states(type_idx=2, n_type=self.params["n_obs"]) if self.params["n_obs"] > 0 else None
        time = graph.env_states.time
        next_time = time + self.dt

        # calculate next graph
        action = self.clip_action(action)
        next_rl_agent_states = self.agent_step_euler(rl_agent_states, action)
        next_non_rl_agent_states = self.non_rl_agent_trajectory(next_time) ################

        # jax.debug.print("next_non_rl_agent_states: {}", next_non_rl_agent_states)
        # jax.debug.print("non_rl_agent_states: {}", non_rl_agent_states)

        next_env_state = MPEEnvState(next_rl_agent_states, next_non_rl_agent_states, obstacles, next_time)
        info = {}

        # the episode ends when reaching max_episode_steps
        done = jnp.array(False)

        # calculate reward and cost
        reward = self.get_reward(graph, action)
        cost = self.get_cost(graph)

        return self.get_graph(next_env_state), reward, cost, done, info

    def get_reward(self, graph: MPEEnvGraphsTuple, action: Action) -> Reward:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        non_rl_agent_states = graph.type_states(type_idx=1, n_type=self.non_rl_agents)
        obstacles = graph.type_states(type_idx=2, n_type=self.params["n_obs"]) if self.params["n_obs"] > 0 else None

        agent_fov_range = 1.0  # 1.0 meters

        ################## visibility check ##################
        # RL and non-RL agent positions
        non_rl_agent_pos = non_rl_agent_states[:, :2]  # [n_non_rl, 2]
        agent_pos = agent_states[:, :2]                # [n_rl, 2]
        pos_diff = non_rl_agent_pos[:, None, :] - agent_pos[None, :, :]  # [n_non_rl, n_rl, 2]

        # Visibility by distance
        is_visible = jnp.linalg.norm(pos_diff, axis=-1) < agent_fov_range  # [n_non_rl, n_rl]
        # jax.debug.print("is_visible: {}",is_visible)
        # jax.debug.print("is_visible.shape: {}",is_visible.shape)

        # Obstacle blocking logic (projection-based robust check)
        obs_pos = obstacles[:, :2]  # [n_obs, 2]
        # For each (non_rl, rl, obs) triple
        # a = agent_pos, b = non_rl_agent_pos, o = obs_pos
        a = agent_pos[None, :, :]      # [1, n_rl, 2]
        b = non_rl_agent_pos[:, None, :]  # [n_non_rl, 1, 2]
        o = obs_pos[:, None, None, :]  # [n_obs, 1, 1, 2]
        # Compute vectors
        ba = b - a                     # [n_non_rl, n_rl, 2]
        oa = o - a                     # [n_obs, n_rl, 2]
        ba_norm_sq = jnp.sum(ba ** 2, axis=-1, keepdims=True)  # [n_non_rl, n_rl, 1]
        #Broadcast oa and ba to [n_obs, n_non_rl, n_rl, 2]
        oa = oa[:, None, :, :]  # [n_obs, 1, n_rl, 2]
        ba = ba[None, :, :, :]  # [1, n_non_rl, n_rl, 2]
        ba_norm_sq = ba_norm_sq[None, :, :, :]  # [1, n_non_rl, n_rl, 1]
        t = jnp.sum(oa * ba, axis=-1, keepdims=True) / (ba_norm_sq + 1e-8)  # [n_obs, n_non_rl, n_rl, 1]
        obs_in_segment = jnp.logical_and(t >= 0.0, t <= 1.0)  # [n_obs, n_non_rl, n_rl, 1]
        obs_in_segment = obs_in_segment.squeeze(-1)           # [n_obs, n_non_rl, n_rl]

        # Distance from obstacle to line segment
        # For each obs, non_rl, rl: compute cross product and normalize
        # Expand obs_pos for broadcasting
        obs_pos_exp = obs_pos[:, None, None, :]  # [n_obs, 1, 1, 2]
        pos_diff_exp = pos_diff[None, :, :, :]   # [1, n_non_rl, n_rl, 2]
        obs_diff = obs_pos_exp - agent_pos[None, None, :, :]  # [n_obs, 1, n_rl, 2]
        obs_diff = obs_diff[:, jnp.newaxis, :, :, :]  # [n_obs, 1, 1, n_rl, 2]
        pos_diff_exp = pos_diff_exp[:, :, :, :]  # [1, n_non_rl, n_rl, 2]  
        # Compute cross product for each obs, non_rl, rl
        cross = pos_diff_exp[..., 0] * obs_diff[..., 1] - pos_diff_exp[..., 1] * obs_diff[..., 0]  # [n_obs, n_non_rl, n_rl]
        norm = jnp.linalg.norm(pos_diff_exp, axis=-1)
        obs_to_line = jnp.abs(cross) / (norm + 1e-8) < self.params["obs_radius"]  # [n_obs, n_non_rl, n_rl]

        # Obstacle blocks sight if both conditions are met
        obs_blocking = jnp.logical_and(obs_to_line, obs_in_segment)  # [n_obs, n_non_rl, n_rl]
        blocked = jnp.any(obs_blocking, axis=0)  # [n_non_rl, n_rl]
        blocked = blocked.squeeze()

        # Mask visibility
        is_visible = jnp.logical_and(is_visible, jnp.logical_not(blocked))
        ################## visibility check ##################

        # Team visibility: at least one RL agent can see at least one non-RL agent
        is_visible_team = jnp.any(is_visible)  # scalar

        # Minimum distance to any non-RL agent for any RL agent
        min_dist = jnp.linalg.norm(pos_diff, axis=-1).min(axis=0)  # [n_rl]
        # jax.debug.print("min_dist: {}",min_dist)
        # jax.debug.print("min_dist.shape: {}",min_dist.shape)
        min_dist = jnp.min(min_dist)  # scalar

        # Reward logic
        reward = jax.lax.cond(
            is_visible_team,
            lambda _: jnp.array(0.0, dtype=jnp.float32),
            lambda _: jnp.array(-0.1 - 0.1 * jnp.abs(min_dist - agent_fov_range), dtype=jnp.float32), #########
            operand=None
        )
        # jax.debug.print("reward: {}",reward)
        # jax.debug.print("\n\n")
        # Action penalty
        reward -= (jnp.linalg.norm(action, axis=1) ** 2).mean() * 0.0001
        return reward

    def get_cost(self, graph: MPEEnvGraphsTuple) -> Cost:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        non_rl_agent_states = graph.type_states(type_idx=1, n_type=self.non_rl_agents)
        obstacles = graph.type_states(type_idx=2, n_type=self.params["n_obs"])[:, :2]

        # collision between agents
        agent_pos = agent_states[:, :2]
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6
        min_dist = jnp.min(dist, axis=1)
        agent_cost: Array = self.params["car_radius"] * 2 - min_dist

        # collision between agents and non_rl_agents
        non_rl_agent_pos = non_rl_agent_states[:, :2]
        # add virtual position 'in front' and 'behind' the non-RL agent
        non_rl_agent_pos = jnp.concatenate([
            non_rl_agent_pos,
            non_rl_agent_pos + non_rl_agent_states[:,2:]/jnp.linalg.norm(non_rl_agent_states[:,2:], axis=-1, keepdims=True) * self.params["obs_radius"]/3*4,  # in front
            non_rl_agent_pos - non_rl_agent_states[:,2:]/jnp.linalg.norm(non_rl_agent_states[:,2:], axis=-1, keepdims=True) * self.params["obs_radius"]/3*4  # behind
        ], axis=0)
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(non_rl_agent_pos, 0), axis=-1)
        min_dist = jnp.min(dist, axis=1)
        target_cost: Array = self.params["car_radius"] + self.params["obs_radius"] - min_dist

        # collision between agents and obstacles
        if self.params["n_obs"] == 0:
            obs_cost = jnp.zeros(self.num_agents)
        else:
            dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(obstacles, 0), axis=-1)
            min_dist = jnp.min(dist, axis=1)
            obs_cost: Array = self.params["car_radius"] + self.params["obs_radius"] - min_dist

        cost = jnp.concatenate([agent_cost[:, None], target_cost[:, None], obs_cost[:, None]], axis=1)
        assert cost.shape == (self.num_agents, self.n_cost)

        # add margin
        eps = 0.5
        cost = jnp.where(cost <= 0.0, cost - eps, cost + eps)
        cost = jnp.clip(cost, a_min=-1.0)

        return cost

    def render_video(
            self,
            rollout: Rollout,
            video_path: pathlib.Path,
            Ta_is_unsafe=None,
            viz_opts: dict = None,
            n_goal: int = None,
            **kwargs
    ) -> None:
        n_goal = self.non_rl_agents

        ax: Axes
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=100)
        ax.set_xlim(0., self.area_size)
        ax.set_ylim(0., self.area_size)
        ax.set(aspect="equal")
        plt.axis("off")
        if viz_opts is None:
            viz_opts = {}

        # plot the first frame
        T_graph = rollout.graph
        graph0 = tree_index(T_graph, 0)

        agent_color = "#0068ff"
        goal_color = "#2fdd00"
        obs_color = "#8a0000"
        edge_goal_color = goal_color

        # plot obstacles
        if self.params["n_obs"] > 0:
            obs_pos = graph0.type_states(type_idx=MPEInspect_Slow.OBS, n_type=self.params["n_obs"])[:, :2]  # [n_obs, 2]
            obs_plots = [plt.Circle((float(obs_pos[i, 0]), float(obs_pos[i, 1])), self.params["obs_radius"],
                                    color=obs_color) for i in range(len(obs_pos))]
            obs_col = MutablePatchCollection(obs_plots, match_original=True, zorder=5)
            ax.add_collection(obs_col)

        # plot agents
        n_pos = np.array(graph0.type_states(type_idx=MPEInspect_Slow.AGENT, n_type=self.num_agents)[:, :2]).astype(np.float32)
        agent_circs = [
            plt.Circle((float(n_pos[i, 0]), float(n_pos[i, 1])), self.params["car_radius"],
                       color=agent_color, linewidth=0.0)
            for i in range(self.num_agents)
        ]
        agent_col = MutablePatchCollection(agent_circs, match_original=True, zorder= 6)
        ax.add_collection(agent_col)    

        # plot goals
        goal_pos = np.array(graph0.type_states(type_idx=MPEInspect_Slow.GOAL, n_type=n_goal)[:, :2]).astype(np.float32)
        goal_circs = [
            plt.Circle((float(goal_pos[i, 0]), float(goal_pos[i, 1])), self.params["car_radius"],
                       color=goal_color, linewidth=0.0)
            for i in range(n_goal)
        ]
        goal_col = MutablePatchCollection(goal_circs, match_original=True, zorder= 6)
        ax.add_collection(goal_col)

        #plot sight range for rl_agents, fan shaped from rl_agent_pos_t to goal_pos_t
        rl_agent_states = graph0.type_states(type_idx=MPEInspect_Slow.AGENT, n_type=self.num_agents)
        rl_agent_pos = rl_agent_states[:, :2]
        rl_agent_sight_range = 1.0
        rl_agent_sight_angle = np.pi/3
        sight_range_polys = []
        for i in range(self.num_agents):
            agent_pos = np.array(rl_agent_pos[i])
            goal_pos_i = np.array(goal_pos[0])
            direction = np.arctan2(goal_pos_i[1] - agent_pos[1], goal_pos_i[0] - agent_pos[0])
            sector_pts = self.get_sector_points(agent_pos, direction, rl_agent_sight_range, rl_agent_sight_angle)
            poly = Polygon(sector_pts, color=agent_color, alpha=0.2)
            sight_range_polys.append(poly)
        sight_range_col = MutablePatchCollection(sight_range_polys, match_original=True, zorder=6)
        ax.add_collection(sight_range_col)
        # angle_range = jnp.linspace(jnp.atan2(goal_pos[:,1] - rl_agent_pos[:,1], goal_pos[:,0] - rl_agent_pos[:,0])-rl_agent_sight_angle/2, jnp.atan2(goal_pos[:,1] - rl_agent_pos[:,1], goal_pos[:,0] - rl_agent_pos[:,0])+rl_agent_sight_angle/2, 50)
        # arc_x = rl_agent_pos[:,0] + rl_agent_sight_range * jnp.cos(angle_range)
        # arc_y = rl_agent_pos[:,1] + rl_agent_sight_range * jnp.sin(angle_range)
        # sector_x = jnp.concatenate([rl_agent_pos[:,0].reshape(-1,self.num_agents), arc_x, rl_agent_pos[:,0].reshape(-1,self.num_agents)],axis=0).flatten()
        # sector_y = jnp.concatenate([rl_agent_pos[:,1].reshape(-1,self.num_agents), arc_y, rl_agent_pos[:,1].reshape(-1,self.num_agents)],axis=0).flatten()
        # ax.fill(sector_x, sector_y, color=agent_color, alpha=0.2)
        # #ax is drawing all timestep's sight range, so we need to clear the previous sight range

        #plot goal-reach-avoid-region
        non_rl_agent_states = graph0.type_states(type_idx=MPEInspect_Slow.GOAL, n_type=n_goal * 3)
        non_rl_agent_pos = non_rl_agent_states[:, :2]
        goal_reach_avoid_region = jnp.concatenate([
            non_rl_agent_pos,
            non_rl_agent_pos + non_rl_agent_states[:,2:]/jnp.linalg.norm(non_rl_agent_states[:,2:], axis=-1, keepdims=True) * self.params["obs_radius"]/3*4,  # in front
            non_rl_agent_pos - non_rl_agent_states[:,2:]/jnp.linalg.norm(non_rl_agent_states[:,2:], axis=-1, keepdims=True) * self.params["obs_radius"]/3*4  # behind
        ], axis=0)
        goal_reach_avoid_region_circs = [
            plt.Circle((float(goal_reach_avoid_region[i, 0]), float(goal_reach_avoid_region[i, 1])), self.params["obs_radius"],
                       color="#40ee11", linewidth=0.0, alpha=0.2)
            for i in range(n_goal * 3)
        ]
        goal_reach_avoid_region_col = MutablePatchCollection(goal_reach_avoid_region_circs, match_original=True, zorder= 6)
        ax.add_collection(goal_reach_avoid_region_col)

        # plot edges
        all_pos = graph0.states[:, :2]
        edge_index = np.stack([graph0.senders, graph0.receivers], axis=0)
        is_pad = np.any(edge_index == self.num_agents + n_goal + self.params["n_obs"], axis=0)
        e_edge_index = edge_index[:, ~is_pad]
        e_start, e_end = all_pos[e_edge_index[0, :]], all_pos[e_edge_index[1, :]]
        e_lines = np.stack([e_start, e_end], axis=1)  # (e, n_pts, dim)
        e_is_goal = (self.num_agents <= graph0.senders) & (graph0.senders < self.num_agents + n_goal)
        e_is_goal = e_is_goal[~is_pad]
        e_colors = [edge_goal_color if e_is_goal[ii] else "0.2" for ii in range(len(e_start))]
        edge_col = LineCollection(e_lines, colors=e_colors, linewidths=2, alpha=0.5, zorder=3)
        ax.add_collection(edge_col)

        # texts
        text_font_opts = dict(
            size=16,
            color="k",
            family="sans-serif",
            weight="normal",
            transform=ax.transAxes,
        )
        cost_text = ax.text(0.02, 1.00, "Cost: 1.0\nReward: 1.0", va="bottom", **text_font_opts)
        if Ta_is_unsafe is not None:
            safe_text = [ax.text(0.99, 1.00, "Unsafe: {}", va="bottom", ha="right", **text_font_opts)]
        kk_text = ax.text(0.99, 1.04, "kk=0", va="bottom", ha="right", **text_font_opts)
        z_text = ax.text(0.5, 1.04, "z: []", va="bottom", ha="center", **text_font_opts)

        # add agent labels
        label_font_opts = dict(
            size=20,
            color="k",
            family="sans-serif",
            weight="normal",
            ha="center",
            va="center",
            transform=ax.transData,
            clip_on=True,
            zorder=7,
        )
        agent_labels = [ax.text(float(n_pos[ii, 0]), float(n_pos[ii, 1]), f"{ii}", **label_font_opts)
                        for ii in range(self.num_agents)]

        if "Vh" in viz_opts:
            Vh_text = ax.text(0.99, 0.99, "Vh: []", va="top", ha="right", **text_font_opts)

        # init function for animation
        def init_fn() -> list[plt.Artist]:
            return [agent_col, edge_col, *agent_labels, cost_text, *safe_text, kk_text]

        def update(kk: int) -> list[plt.Artist]:
            graph = tree_index(T_graph, kk)
            n_pos_t = graph.states[:-1, :2]

            # update agent positions
            for ii in range(self.num_agents):
                agent_circs[ii].set_center(tuple(n_pos_t[ii]))

            # update goal positions
            goal_pos_t = np.array(graph.type_states(type_idx=MPEInspect_Slow.GOAL, n_type=n_goal)[:, :2]).astype(np.float32)
            for ii in range(n_goal):
                goal_circs[ii].set_center(tuple(goal_pos_t[ii]))
            # update goal color based on reward
            reward_t = rollout.rewards[kk] if kk < len(rollout.rewards) else 0.0
            goal_color_green = "#2fdd00"
            goal_color_yellow = "#ffe100"
            color_to_set = goal_color_yellow if reward_t > -0.5 else goal_color_green
            for ii in range(n_goal):
                goal_circs[ii].set_facecolor(color_to_set)

            #make sight range for rl_agents
            rl_agent_states_t = np.array(graph.type_states(type_idx=MPEInspect_Slow.AGENT, n_type=self.num_agents))
            rl_agent_pos_t = rl_agent_states_t[:, :2]
            goal_pos_t = np.array(graph.type_states(type_idx=MPEInspect_Slow.GOAL, n_type=n_goal)[:, :2]).astype(np.float32)
            for i in range(self.num_agents):
                agent_pos = np.array(rl_agent_pos_t[i])
                goal_pos_i = np.array(goal_pos_t[0])
                direction = np.arctan2(goal_pos_i[1] - agent_pos[1], goal_pos_i[0] - agent_pos[0])
                sector_pts = self.get_sector_points(agent_pos, direction, 1.0, np.pi/3)
                sight_range_polys[i].set_xy(sector_pts)
            
            #update goal-reach-avoid-region
            non_rl_agent_states_t = np.array(graph.type_states(type_idx=MPEInspect_Slow.GOAL, n_type=n_goal))
            non_rl_agent_pos_t = non_rl_agent_states_t[:, :2]
            non_rl_agent_updates = np.concatenate([
                non_rl_agent_pos_t,
                non_rl_agent_pos_t + non_rl_agent_states_t[:,2:]/np.linalg.norm(non_rl_agent_states_t[:,2:], axis=-1, keepdims=True) * self.params["obs_radius"]/3*4,  # in front
                non_rl_agent_pos_t - non_rl_agent_states_t[:,2:]/np.linalg.norm(non_rl_agent_states_t[:,2:], axis=-1, keepdims=True) * self.params["obs_radius"]/3*4  # behind
            ], axis=0)
            for ii in range(n_goal * 3):
                goal_reach_avoid_region_circs[ii].set_center(tuple(non_rl_agent_updates[ii]))
            
            #update sight range for rl_agents
            # rl_agent_pos_t = rl_agent_states_t[:, :2]
            # angle_range_t = jnp.linspace(jnp.atan2(goal_pos_t[:,1] - rl_agent_pos_t[:,1], goal_pos_t[:,0] - rl_agent_pos_t[:,0])-rl_agent_sight_angle/2, jnp.atan2(goal_pos_t[:,1] - rl_agent_pos_t[:,1], goal_pos_t[:,0] - rl_agent_pos_t[:,0])+rl_agent_sight_angle/2, 50)
            # arc_x_t = rl_agent_pos_t[:,0] + rl_agent_sight_range * jnp.cos(angle_range_t)
            # arc_y_t = rl_agent_pos_t[:,1] + rl_agent_sight_range * jnp.sin(angle_range_t)
            # sector_x_t = jnp.concatenate([rl_agent_pos_t[:,0].reshape(-1,self.num_agents), arc_x_t, rl_agent_pos_t[:,0].reshape(-1,self.num_agents)],axis=0).flatten()
            # sector_y_t = jnp.concatenate([rl_agent_pos_t[:,1].reshape(-1,self.num_agents), arc_y_t, rl_agent_pos_t[:,1].reshape(-1,self.num_agents)],axis=0).flatten()
            # ax.fill(sector_x_t, sector_y_t, color=agent_color, alpha=0.2)


            # update edges
            e_edge_index_t = np.stack([graph.senders, graph.receivers], axis=0)
            is_pad_t = np.any(e_edge_index_t == self.num_agents + n_goal + self.params["n_obs"], axis=0)
            e_edge_index_t = e_edge_index_t[:, ~is_pad_t]
            e_start_t, e_end_t = n_pos_t[e_edge_index_t[0, :]], n_pos_t[e_edge_index_t[1, :]]
            e_is_goal_t = (self.num_agents <= graph.senders) & (graph.senders < self.num_agents + n_goal)
            e_is_goal_t = e_is_goal_t[~is_pad_t]
            e_colors_t = [edge_goal_color if e_is_goal_t[ii] else "0.2" for ii in range(len(e_start_t))]
            e_lines_t = np.stack([e_start_t, e_end_t], axis=1)
            edge_col.set_segments(e_lines_t)
            edge_col.set_colors(e_colors_t)

            # update agent labels
            for ii in range(self.num_agents):
                agent_labels[ii].set_position(n_pos_t[ii])

            # update cost and safe labels
            if kk < len(rollout.costs):
                all_costs = ""
                for i_cost in range(rollout.costs[kk].shape[1]):
                    all_costs += f"    {self.cost_components[i_cost]}: {rollout.costs[kk][:, i_cost].max():5.4f}\n"
                all_costs = all_costs[:-3]
                cost_text.set_text(f"Cost:\n{all_costs}\nReward: {rollout.rewards[kk]:5.4f}")
            else:
                cost_text.set_text("")
            if kk < len(Ta_is_unsafe):
                a_is_unsafe = Ta_is_unsafe[kk]
                unsafe_idx = np.where(a_is_unsafe)[0]
                safe_text[0].set_text("Unsafe: {}".format(unsafe_idx))
            else:
                safe_text[0].set_text("Unsafe: {}")

            kk_text.set_text("kk={:04}".format(kk))

            # Update the z text.
            z_text.set_text(f"z: {rollout.zs[kk]}")

            if "Vh" in viz_opts:
                Vh_text.set_text(f"Vh: {viz_opts['Vh'][kk]}")

            return [agent_col, edge_col, *agent_labels, cost_text, *safe_text, kk_text]

        fps = 30.0
        spf = 1 / fps
        mspf = 1_000 * spf
        anim_T = len(T_graph.n_node)
        ani = FuncAnimation(fig, update, frames=anim_T, init_func=init_fn, interval=mspf, blit=True)
        save_anim(ani, video_path)

    def get_sector_points(self, center, direction, radius, angle, n_points=30):
        """
        center: (2,) array, agent position
        direction: float, angle in radians (direction to goal)
        radius: float, sight range
        angle: float, total angle of the sector (radians)
        n_points: int, number of points along the arc
        Returns: (n_points+2, 2) array of (x, y) points
        """
        angles = np.linspace(direction - angle/2, direction + angle/2, n_points)
        arc = np.stack([
            center[0] + radius * np.cos(angles),
            center[1] + radius * np.sin(angles)
        ], axis=1)
        # Sector: center + arc + center to close
        return np.vstack([center, arc, center])

    def edge_blocks(self, state: MPEEnvState) -> list[EdgeBlock]:
        # agent - agent connection
        agent_pos = state.agent[:, :2]
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        state_diff = state.agent[:, None, :] - state.agent[None, :, :]
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(dist.shape[1]) * (self._params["comm_radius"] + 1)
        agent_agent_mask = jnp.less(dist, self._params["comm_radius"])
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(state_diff, agent_agent_mask, id_agent, id_agent)

        # agent - goal connection
        id_goal = jnp.arange(self.num_agents, self.num_agents + self.non_rl_agents)
        agent_goal_mask = jnp.ones((self.num_agents, self.non_rl_agents))
        agent_goal_feats = state.agent[:, None, :] - state.goal[None, :, :]
        agent_goal_edges = EdgeBlock(
            agent_goal_feats, agent_goal_mask, id_agent, id_goal
        )

        # agent - obs connection
        if self._params["n_obs"] == 0:
            return [agent_agent_edges, agent_goal_edges]
        obs_pos = state.obs[:, :2]
        poss_diff = agent_pos[:, None, :] - obs_pos[None, :, :]
        dist = jnp.linalg.norm(poss_diff, axis=-1)
        agent_obs_mask = jnp.less(dist, self._params["comm_radius"])
        id_obs = jnp.arange(self._params["n_obs"]) + self.num_agents * 2
        state_diff = state.agent[:, None, :] - state.obs[None, :, :]
        agent_obs_edges = EdgeBlock(state_diff, agent_obs_mask, id_agent, id_obs)

        return [agent_agent_edges, agent_goal_edges, agent_obs_edges]        

    def get_graph(self, env_state: MPEEnvState) -> MPEEnvGraphsTuple:
        # node features
        # states
        node_feats = jnp.zeros((self.num_agents + self.non_rl_agents + self.params["n_obs"], self.node_dim))
        node_feats = node_feats.at[:self.num_agents, :self.state_dim].set(env_state.agent)
        node_feats = node_feats.at[self.num_agents: self.num_agents + self.non_rl_agents, :self.state_dim].set(env_state.goal)
        if self.params["n_obs"] > 0:
            node_feats = node_feats.at[self.num_agents + self.non_rl_agents:, :self.state_dim].set(env_state.obs)

        # indicators
        node_feats = node_feats.at[:self.num_agents, 6].set(1.0)
        node_feats = node_feats.at[self.num_agents: self.num_agents + self.non_rl_agents, 5].set(1.0)
        if self.params["n_obs"] > 0:
            node_feats = node_feats.at[self.num_agents + self.non_rl_agents:, 4].set(1.0)

        # node type
        node_type = -jnp.ones((self.num_agents + self.non_rl_agents + self.params["n_obs"],), dtype=jnp.int32)
        node_type = node_type.at[:self.num_agents].set(MPEInspect_Slow.AGENT)
        node_type = node_type.at[self.num_agents: self.num_agents + self.non_rl_agents].set(MPEInspect_Slow.GOAL)
        if self.params["n_obs"] > 0:
            node_type = node_type.at[self.num_agents + self.non_rl_agents:].set(MPEInspect_Slow.OBS)

        # edges
        edge_blocks = self.edge_blocks(env_state)

        # create graph
        states = jnp.concatenate([env_state.agent, env_state.goal], axis=0)
        if self.params["n_obs"] > 0:
            states = jnp.concatenate([states, env_state.obs], axis=0)
        return GetGraph(node_feats, node_type, edge_blocks, env_state, states).to_padded()

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        lower_lim = jnp.array([0.0, 0.0, -0.7, -0.7])
        upper_lim = jnp.array([self.area_size, self.area_size, 0.7, 0.7])
        return lower_lim, upper_lim

    def action_lim(self) -> Tuple[Action, Action]:
        lower_lim = jnp.ones(2) * -0.7
        upper_lim = jnp.ones(2) * 0.7
        return lower_lim, upper_lim

    @ft.partial(jax.jit, static_argnums=(0,))
    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        cost = self.get_cost(graph)
        return jnp.any(cost >= 0.0, axis=-1)
