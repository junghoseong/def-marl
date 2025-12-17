import argparse
import datetime
import functools as ft
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import pathlib
import ipdb
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import yaml

from defmarl.algo import make_algo
from defmarl.env import make_env
from defmarl.trainer.data import Rollout
from defmarl.trainer.utils import test_rollout
from defmarl.utils.utils import jax_jit_np, jax_vmap
from defmarl.utils.utils import tree_index


def test(args):
    print(f"> Running test.py {args}")

    stamp_str = datetime.datetime.now().strftime("%m%d-%H%M")

    # set up environment variables and seed
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    if args.debug:
        jax.config.update("jax_disable_jit", True)
    np.random.seed(args.seed)

    # load config
    if args.path is not None:
        with open(os.path.join(args.path, "config.yaml"), "r") as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)

    # create environments
    num_agents = config.num_agents if args.num_agents is None else args.num_agents
    env = make_env(
        env_id=config.env if args.env is None else args.env,
        num_agents=num_agents,
        num_obs=config.obs if args.obs is None else args.obs,
        max_step=args.max_step,
        full_observation=args.full_observation,
        area_size=config.area_size if args.area_size is None else args.area_size,
    )

    path = args.path
    model_path = os.path.join(path, "models")
    if args.step is None:
        models = os.listdir(model_path)
        step = max([int(model) for model in models if model.isdigit()])
    else:
        step = args.step
    print("step: ", step)

    algo = make_algo(
        algo=config.algo,
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=env.num_agents,
        cost_weight=config.cost_weight,
        actor_gnn_layers=config.gnn_layers,
        critic_gnn_layers=config.gnn_layers,
        Vh_gnn_layers=config.Vh_gnn_layers if hasattr(config, "Vh_gnn_layers") else 1,
        lr_actor=config.lr_actor,
        lr_cbf=config.lr_critic,
        max_grad_norm=2.0,
        seed=config.seed,
        use_rnn=config.use_rnn,
        rnn_layers=config.rnn_layers,
        use_lstm=config.use_lstm,
    )
    algo.load(model_path, step)
    if args.stochastic:
        def act_fn(x, z, rnn_state, key):
            action, _, new_rnn_state = algo.step(x, z, rnn_state, key)
            return action, new_rnn_state
        act_fn = jax.jit(act_fn)
    else:
        act_fn = algo.act
    if hasattr(algo, "init_Vh_rnn_state"):
        init_Vh_rnn_state = algo.init_Vh_rnn_state
    else:
        init_Vh_rnn_state = None

    init_rnn_state = algo.init_rnn_state  # <-- FIX: define before z_settings

    # --- NEW: Prepare z settings for evaluation ---
    z_settings = {
        "opt_z": algo.get_opt_z if hasattr(algo, "get_opt_z") else None,
        # "z_min": lambda graph, value_rnn_state: (
        #     jnp.array([[-env.reward_max]]).repeat(env.num_agents, axis=0), value_rnn_state),
        # "z_max": lambda graph, value_rnn_state: (
        #     jnp.array([[-env.reward_min]]).repeat(env.num_agents, axis=0), value_rnn_state),
    }

    test_key = jr.PRNGKey(args.seed)
    test_keys = jr.split(test_key, 1_000)[: args.epi]
    test_keys = test_keys[args.offset:]

    # --- NEW: Loop over z settings ---
    for z_name, z_fn in z_settings.items():
        # Prepare rollout and video output directory
        videos_dir = pathlib.Path(path) / "videos" / f"{step}" / z_name
        videos_dir.mkdir(exist_ok=True, parents=True)

        # Prepare act_fn for this z setting
        if args.stochastic:
            def act_fn(x, z, rnn_state, key):
                action, _, new_rnn_state = algo.step(x, z, rnn_state, key)
                return action, new_rnn_state
            act_fn = jax.jit(act_fn)
        else:
            act_fn = algo.act
        # z_fn is already set by the loop
        rollout_fn = ft.partial(test_rollout,
                                env,
                                act_fn,
                                init_rnn_state,
                                init_Vh_rnn_state=init_Vh_rnn_state,
                                z_fn=z_fn,
                                stochastic=args.stochastic)
        rollout_fn = jax_jit_np(rollout_fn)
        is_unsafe_fn = jax_jit_np(jax_vmap(env.unsafe_mask))

        rewards = []
        costs = []
        is_unsafes = []
        rates = []
        rollouts = []

        for i_epi in range(args.epi):
            key_x0, _ = jr.split(test_keys[i_epi], 2)

            rollout: Rollout = rollout_fn(key_x0)
            is_unsafes.append(is_unsafe_fn(rollout.graph))

            epi_reward = rollout.rewards.sum()
            epi_cost = rollout.costs.max()
            rewards.append(epi_reward)
            costs.append(epi_cost)
            rollouts.append(rollout)

            safe_rate = 1 - is_unsafes[-1].max(axis=0).mean()
            print(f"[z={z_name}] epi: {i_epi}, reward: {epi_reward:.3f}, cost: {epi_cost:.3f}, safe rate: {safe_rate * 100:.3f}%")

            rates.append(np.array(safe_rate))

        is_unsafe = np.max(np.stack(is_unsafes), axis=1)
        safe_mean, safe_std = (1 - is_unsafe).mean(), (1 - is_unsafe).std()

        print(
            f"[z={z_name}] reward: {np.mean(rewards):.3f}, std: {np.std(rewards):.3f}, min/max reward: {np.min(rewards):.3f}/{np.max(rewards):.3f}, "
            f"cost: {np.mean(costs):.3f} min/max cost: {np.min(costs):.3f}/{np.max(costs):.3f}, "
            f"safe_rate: {safe_mean * 100:.3f}%, std: {safe_std * 100:.3f}%"
        )

        # save results
        if args.log:
            with open(os.path.join(path, f"test_log_{z_name}.csv"), "a") as f:
                f.write(f"{env.num_agents},{args.epi},{env.max_episode_steps},"
                        f"{env.area_size},{env.params['n_obs']},"
                        f"{safe_mean * 100:.3f},{safe_std * 100:.3f}\n")

        # make video
        if args.no_video:
            continue

        for ii, (rollout, Ta_is_unsafe) in enumerate(zip(rollouts, is_unsafes)):
            safe_rate = rates[ii] * 100
            video_name = f"n{num_agents}_epi{ii:02}_reward{rewards[ii]:.3f}_cost{costs[ii]:.3f}_sr{safe_rate:.0f}"
            viz_opts = {}
            video_path = videos_dir / f"{stamp_str}_{video_name}.mp4"
            env.render_video(rollout, video_path, Ta_is_unsafe, viz_opts, dpi=args.dpi)
            # CSV export disabled by request
            # # --- NEW: Save RL agent and goal states as CSV ---
            # # RL agents: state and action per timestep, per agent
            # rl_agent_states = []  # List of [T, n_agents, state_dim]
            # rl_agent_actions = [] # List of [T, n_agents, action_dim]
            # for t in range(len(rollout.graph.n_node)):
            #     graph_t = tree_index(rollout.graph, t)
            #     # Get RL agent states at timestep t
            #     rl_states_t = np.array(graph_t.type_states(type_idx=0, n_type=env.num_agents))
            #     rl_agent_states.append(rl_states_t)
            #     # Get RL agent actions at timestep t
            #     if hasattr(rollout, 'actions'):
            #         rl_actions_t = np.array(rollout.actions[t])
            #     else:
            #         rl_actions_t = np.full((env.num_agents, env.action_dim), np.nan)
            #     rl_agent_actions.append(rl_actions_t)
            # rl_agent_states = np.stack(rl_agent_states)  # [T, n_agents, state_dim]
            # rl_agent_actions = np.stack(rl_agent_actions)  # [T, n_agents, action_dim]
            # for agent_idx in range(env.num_agents):
            #     agent_csv_path = videos_dir / f"{stamp_str}_{video_name}_rl_agent{agent_idx}.csv"
            #     with open(agent_csv_path, 'w') as f:
            #         header = ','.join([f'state_{i}' for i in range(rl_agent_states.shape[2])] + [f'action_{i}' for i in range(rl_agent_actions.shape[2])])
            #         f.write('t,' + header + '\n')
            #         for t in range(rl_agent_states.shape[0]):
            #             state_vals = ','.join(map(str, rl_agent_states[t, agent_idx]))
            #             action_vals = ','.join(map(str, rl_agent_actions[t, agent_idx]))
            #             f.write(f'{t},{state_vals},{action_vals}\n')
            # # Non-RL agents (goals): state per timestep, per agent
            # n_goal = env.non_rl_agents if hasattr(env, 'non_rl_agents') else 1
            # goal_states = []
            # for t in range(len(rollout.graph.n_node)):
            #     graph_t = tree_index(rollout.graph, t)
            #     goal_states_t = np.array(graph_t.type_states(type_idx=1, n_type=n_goal))
            #     goal_states.append(goal_states_t)
            # goal_states = np.stack(goal_states)  # [T, n_goal, state_dim]
            # for goal_idx in range(n_goal):
            #     goal_csv_path = videos_dir / f"{stamp_str}_{video_name}_goal{goal_idx}.csv"
            #     with open(goal_csv_path, 'w') as f:
            #         header = ','.join([f'state_{i}' for i in range(goal_states.shape[2])])
            #         f.write('t,' + header + '\n')
            #         for t in range(goal_states.shape[0]):
            #             state_vals = ','.join(map(str, goal_states[t, goal_idx]))
            #             f.write(f'{t},{state_vals}\n')
            # # --- END NEW ---
        print(f"[z={z_name}] Videos saved to {videos_dir}")

    # --- END NEW CODE ---
    # Remove the old single-z rollout/video logic below this point
    return  # Prevent running the old code after the new loop


def main():
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument("--path", type=str, required=True)

    # optional arguments
    parser.add_argument("--epi", type=int, default=5)
    parser.add_argument("--no-video", action="store_true", default=False)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("-n", "--num-agents", type=int, default=None)
    parser.add_argument("--obs", type=int, default=None)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--full-observation", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--stochastic", action="store_true", default=False)
    parser.add_argument("--log", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("-z", type=str, default=None)
    parser.add_argument("--area-size", type=float, default=None)
    parser.add_argument("--offset", type=int, default=0)

    args = parser.parse_args()
    test(args)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
