import jax.numpy as jnp
import jax.random as jr
import optax
import os
import jax
import functools as ft
import jax.tree_util as jtu
import numpy as np
import pickle

from typing import Optional, Tuple
from flax.training.train_state import TrainState
from jax import lax

from .module.root_finder import RootFinder
from ..utils.typing import Action, IntrinsicReward,Params, PRNGKey, Array, FloatScalar
from ..utils.graph import GraphsTuple
from ..utils.utils import jax_vmap, tree_index, tree_where
from ..trainer.data import Rollout
from ..trainer.utils import has_any_nan_or_inf, compute_norm_and_clip
from ..trainer.utils import rollout as rollout_fn
from ..trainer.utils import rollout_efxplorer
from ..env.base import MultiAgentEnv
from ..algo.module.value import ValueNet
from ..algo.module.policy import PPOPolicy
from .utils import compute_dec_efocp_gae, compute_dec_efocp_V, compute_gae, compute_extrinsic_gae, compute_conservative_exploration_gae, compute_gae_single
from .base import Algorithm


class EFXplorer(Algorithm):
    def __init__(
            self,
            env: MultiAgentEnv,
            node_dim: int,
            edge_dim: int,
            state_dim: int,
            action_dim: int,
            n_agents: int,
            cost_weight: float = 0.,
            actor_gnn_layers: int = 2,
            critic_gnn_layers: int = 2,
            gamma: float = 0.99,
            lr_actor: float = 1e-5,
            lr_critic: float = 1e-5,
            batch_size: int = 8192,
            epoch_ppo: int = 1,
            clip_eps: float = 0.25,
            gae_lambda: float = 0.95,
            coef_ent: float = 1e-2,
            max_grad_norm: float = 2.0,
            seed: int = 0,
            use_rnn: bool = True,
            rnn_layers: int = 1,
            rnn_step: int = 16,
            rollout_length: Optional[int] = None,
            use_lstm: bool = False,
            use_prev_init: bool = False,
            # Intrinsic reward related parameters
            intrinsic_coef: float = 1.0,
            lr_z: float = 1e-7,
            **kwargs
    ):
        super(EFXplorer, self).__init__(
            env=env,
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=action_dim,
            n_agents=n_agents
        )

        # set hyperparameters
        self.cost_weight = cost_weight
        self.actor_gnn_layers = actor_gnn_layers
        self.critic_gnn_layers = critic_gnn_layers
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.batch_size = batch_size
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.gae_lambda = gae_lambda
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        self.rollout_length = rollout_length
        self.use_rnn = use_rnn
        self.rnn_layers = rnn_layers
        self.rnn_step = rnn_step
        self.use_lstm = use_lstm
        self.use_prev_init = use_prev_init
        self.intrinsic_coef = intrinsic_coef
        self.lr_z = lr_z
        
        # set nominal graph for initialization of the neural networks
        nominal_graph = GraphsTuple(
            nodes=jnp.zeros((n_agents, node_dim)),
            edges=jnp.zeros((n_agents, edge_dim)),
            states=jnp.zeros((n_agents, state_dim)),
            n_node=jnp.array(n_agents),
            n_edge=jnp.array(n_agents),
            senders=jnp.arange(n_agents),
            receivers=jnp.arange(n_agents),
            node_type=jnp.zeros((n_agents,)),
            env_states=jnp.zeros((n_agents,)),
        )
        self.nominal_graph = nominal_graph
        # z is shared across agents; keep it 2D for consistent indexing/broadcast
        self.nominal_z = jnp.zeros((1, 1))
        
        # Initialize z_current for budget tracking (initial z = 0)
        self.z_current = jnp.zeros((self.n_agents, 1))

        # set up PPO policy
        self.policy = PPOPolicy(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            n_agents=self.n_agents,
            action_dim=self.action_dim,
            use_rnn=self.use_rnn,
            rnn_layers=self.rnn_layers,
            gnn_layers=self.actor_gnn_layers,
            gnn_out_dim=64,
            use_lstm=self.use_lstm,
        )

        # initialize the rnn state
        key = jr.PRNGKey(seed)
        rnn_state_key, key = jr.split(key)
        rnn_state_key = jr.split(rnn_state_key, self.n_agents)
        init_rnn_state = jax_vmap(self.policy.initialize_carry)(rnn_state_key)  # (n_agents, rnn_state_dim)
        if type(init_rnn_state) is tuple:
            init_rnn_state = jnp.stack(init_rnn_state, axis=1)  # (n_agents, n_carries, rnn_state_dim)
        else:
            init_rnn_state = jnp.expand_dims(init_rnn_state, axis=1)
        # (n_rnn_layers, n_agents, n_carries, rnn_state_dim)
        self.init_rnn_state = init_rnn_state[None, :, :, :].repeat(self.rnn_layers, axis=0)

        # initialize the policy
        policy_key, key = jr.split(key)
        policy_params = self.policy.dist.init(
            policy_key, nominal_graph, self.init_rnn_state, self.n_agents,self.nominal_z.repeat(self.n_agents, axis=0)
        )
        policy_optim = optax.adam(learning_rate=lr_actor)
        self.policy_optim = optax.apply_if_finite(policy_optim, 1_000_000)
        self.policy_train_state = TrainState.create(
            apply_fn=self.policy.sample_action,
            params=policy_params,
            tx=self.policy_optim
        )

        # set up PPO critic
        self.critic = ValueNet(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            n_agents=self.n_agents,
            use_rnn=self.use_rnn,
            rnn_layers=self.rnn_layers,
            gnn_layers=self.critic_gnn_layers,
            gnn_out_dim=64,
            use_lstm=self.use_lstm,
            use_ef=False,
            decompose=False,
        )

        # initialize the rnn state
        rnn_state_key, key = jr.split(key)
        init_Vl_rnn_state = self.critic.initialize_carry(rnn_state_key)  # (rnn_state_dim,)
        if type(init_Vl_rnn_state) is tuple:
            init_Vl_rnn_state = jnp.stack(init_Vl_rnn_state, axis=0)  # (n_carries, rnn_state_dim)
        else:
            init_Vl_rnn_state = init_Vl_rnn_state[None, :]
        # (n_rnn_layers, 1, n_carries, rnn_state_dim)
        self.init_Vl_rnn_state = init_Vl_rnn_state[None, :, :].repeat(self.rnn_layers, axis=0)[:, None, :, :]

        # initialize the critic
        critic_key, key = jr.split(key)
        critic_params = self.critic.net.init(
            critic_key, nominal_graph, self.init_Vl_rnn_state, self.n_agents, self.nominal_z)
        critic_optim = optax.adam(learning_rate=lr_critic)
        self.critic_optim = optax.apply_if_finite(critic_optim, 1_000_000)
        self.critic_train_state = TrainState.create(
            apply_fn=self.critic.get_value,
            params=critic_params,
            tx=self.critic_optim
        )
        
        # set up key
        self.key = key

        # rollout function
        def rollout_fn_single_(cur_params, cur_key, cur_init_graph=None):
            return rollout_efxplorer(self._env,
                              ft.partial(self.step, params=cur_params),
                              self.init_rnn_state,
                              cur_key,
                              self.gamma,
                              self.intrinsic_coef,
                              cur_init_graph)

        def rollout_fn_(cur_params, cur_keys, cur_init_graphs=None):
            return jax.vmap(ft.partial(rollout_fn_single_, cur_params))(cur_keys, cur_init_graphs)

        self.rollout_fn = jax.jit(rollout_fn_)
        
        def get_init_graph_(init_graph_key: PRNGKey, memory: Rollout):
            reset_key, rng_key, idx_key = jr.split(init_graph_key, 3)
            rng = jr.uniform(rng_key, ())
            reset_graph = self._env.reset(reset_key)
            if memory is not None:
                idx = jr.randint(idx_key, (), 0, memory.dones.shape[0])
                memory_graph = tree_index(memory.graph, idx)
                memory_graph = tree_index(memory_graph, -1)
            else:
                return reset_graph
            use_memory = (rng < 0.5) & (memory is not None)
            init_graph = tree_where(use_memory, memory_graph, reset_graph)
            return init_graph
        
        self.get_init_graph = jax.jit(get_init_graph_)
        self.memory = None

    @property
    def config(self) -> dict:
        return {
            'cost_weight': self.cost_weight,
            'actor_gnn_layers': self.actor_gnn_layers,
            'critic_gnn_layers': self.critic_gnn_layers,
            'gamma': self.gamma,
            'rollout_length': self.rollout_length,
            'lr_actor': self.lr_actor,
            'lr_critic': self.lr_critic,
            'batch_size': self.batch_size,
            'epoch_ppo': self.epoch_ppo,
            'clip_eps': self.clip_eps,
            'gae_lambda': self.gae_lambda,
            'coef_ent': self.coef_ent,
            'max_grad_norm': self.max_grad_norm,
            'seed': self.seed,
            'use_rnn': self.use_rnn,
            'rnn_layers': self.rnn_layers,
            'rnn_step': self.rnn_step,
            'use_lstm': self.use_lstm,
            'use_prev_init': self.use_prev_init,
            'intrinsic_coef': self.intrinsic_coef,
            'lr_z': self.lr_z
        }

    @property
    def params(self) -> Params:
        return self.policy_train_state.params

    def act(
            self,
            graph: GraphsTuple,
            z: Array,
            rnn_state: Array,
            params: Optional[Params] = None,
    ) -> [Action, Array]:
        if params is None:
            params = self.params
        # params is the policy parameter tree
        action, rnn_state = self.policy.get_action(params, graph, rnn_state, z)
        return action, rnn_state

    def step(
            self, graph: GraphsTuple, z: Array, rnn_state: Array, key: PRNGKey, params: Optional[Params] = None
    ) -> Tuple[Action, Array, Array]:
        if params is None:
            params = self.params
        # params is the policy parameter tree
        action, log_pi, rnn_state = self.policy_train_state.apply_fn(params, graph, rnn_state, key, z)
        return action, log_pi, rnn_state
    
    def compute_intrinsic_reward(
        self,
        log_pis: Array,
        intrinsic_coef: float = 1.0,
        normalize: bool = True
    ) -> IntrinsicReward:
        intrinsic_rewards = -log_pis * intrinsic_coef
        if normalize:
            mean = intrinsic_rewards.mean()
            std = intrinsic_rewards.std() + 1e-8
            intrinsic_rewards = (intrinsic_rewards - mean) / std
        return intrinsic_rewards

    def collect(self, params: Params, b_key: PRNGKey) -> Rollout:
        if not self.use_prev_init or self.memory is None:
            rollout_result = self.rollout_fn(params, b_key)
            return rollout_result
        else:
            init_rollout_key = jax.vmap(jr.split)(b_key)
            init_key = init_rollout_key[:, 0]
            rollout_key = init_rollout_key[:, 1]
            init_graphs = jax.vmap(ft.partial(self.get_init_graph, memory=self.memory))(init_key)
            
            rollout_result = self.rollout_fn(params, rollout_key, init_graphs)
            return rollout_result

    def update(self, rollout: Rollout, step: int) -> dict:
        key, self.key = jr.split(self.key)

        update_info = {}
        assert rollout.dones.shape[0] * rollout.dones.shape[1] >= self.batch_size
        # jax.debug.print(f"rollout.dones.shape: {rollout.dones.shape}")
        for i_epoch in range(self.epoch_ppo):
            idx = np.arange(rollout.dones.shape[0])
            np.random.shuffle(idx)
            rnn_chunk_ids = jnp.arange(rollout.dones.shape[1])
            # jax.debug.print(f"rnn_chunk_ids: {rnn_chunk_ids}")
            # jax.debug.print(f"rollout.dones.shape[1] // self.rnn_step: {rollout.dones.shape[1] // self.rnn_step}")
            rnn_chunk_ids = jnp.array(jnp.array_split(rnn_chunk_ids, rollout.dones.shape[1] // self.rnn_step))
            batch_idx = jnp.array(jnp.array_split(idx, idx.shape[0] // (self.batch_size // rollout.dones.shape[1])))
            critic_train_state, policy_train_state, update_info = self.update_inner(
                self.critic_train_state,
                self.policy_train_state,
                rollout,
                batch_idx,
                rnn_chunk_ids
            )
            self.critic_train_state = critic_train_state
            self.policy_train_state = policy_train_state
        
        # Update z based on previous rollout's intrinsic reward sum
        z_old = self.z_current
        self.z_current = self.update_z(rollout, z_old)
        
        # Add z update info to logging
        update_info['z/mean'] = self.z_current.mean()
        update_info['z/change'] = (self.z_current - z_old).mean()
        update_info['z/value'] = self.z_current[0, 0]  # Single scalar value (shared across agents)
        
        if self.use_prev_init:
            self.memory = rollout
        return update_info

    def scan_value(
            self,
            rollout: Rollout,
            init_rnn_state_Ve: Array,
            critic_params: Params,
    ) ->Tuple[Tuple[Array, Array], Tuple[Array, Array], Tuple[Array, Array]]:
        """
        Compute extrinsic (Ve) values
        Note: z-conditioning removed to match InforMARL structure
        """
        graphs = rollout.graph  # (T,)
        
        def body_(rnn_state, graph):
            rnn_state_Ve = rnn_state
            # Ve: Global extrinsic value (no z-conditioning)
            value_e, new_rnn_state_Ve = self.critic.get_value(
                critic_params, graph, rnn_state_Ve, None
            )
            return (new_rnn_state_Ve), (value_e, rnn_state_Ve)
        
        # Scan over all timesteps
        (final_rnn_state_Ve), (T_Ve, rnn_states_Ve) = (
            jax.lax.scan(body_, (init_rnn_state_Ve), graphs))
                
        # keep time dimension even when T=1
        T_Ve = jnp.squeeze(T_Ve, axis=-1)  # (T,)
        
        return (T_Ve), (rnn_states_Ve), (final_rnn_state_Ve)

    @ft.partial(jax.jit, static_argnums=(0,))
    def update_inner(
            self,
            critic_train_state: TrainState,
            policy_train_state: TrainState,
            rollout: Rollout,
            batch_idx: Array,
            rnn_chunk_ids: Array
    ) -> Tuple[TrainState, TrainState, dict]:
        # rollout: (n_env, T, n_agent, ...)

        # calculate values and next_values
        """
        Update policy using conservative exploration
        """
        
        # 1. Compute Ve
        bT_Ve, rnn_states_Ve, final_rnn_states_Ve = jax_vmap(
            ft.partial(self.scan_value,
                       init_rnn_state_Ve=self.init_Vl_rnn_state,
                       critic_params=critic_train_state.params)
        )(rollout)  # values: (b, T)

        # 2. Compute Final Ve
        def final_value_fn(graph, rnn_state):
            # No z-conditioning (matching InforMARL)
            return self.critic.get_value(critic_train_state.params, tree_index(graph, -1), rnn_state, None)

        final_Ve, _ = jax.vmap(final_value_fn)(rollout.next_graph, final_rnn_states_Ve)
        final_Ve = jnp.squeeze(final_Ve, axis=-1)  # (b,)
        bTp1_Ve = jnp.concatenate([bT_Ve, final_Ve[:, None]], axis=1) # (b, T+1)
        
        # 3. Team intrinsic reward
        bT_Delta_team_int = rollout.intrinsic_rewards.sum(axis=-1) # (b, T)

        # 4. Ae (extrinsic advantage)
        rollout_rewards = rollout.rewards
        if rollout_rewards.ndim == 2:
            rollout_rewards = rollout_rewards[..., None].repeat(self.n_agents, axis=-1)  # (b, T, a)

        def compute_Aext_single(rewards, values, next_values, dones):
            # rewards: (T, a), values: (T,1) squeezed -> (T,), next_values: (T,), dones: (T,)
            rewards_sum = rewards.sum(axis=-1)  # (T,)
            # Compute targets and raw gaes (without normalization to match InforMARL)
            # Note: compute_gae_single returns normalized gaes, so we compute raw gaes here
            deltas = rewards_sum + self.gamma * next_values.squeeze(-1) * (1 - dones) - values.squeeze(-1)
            gaes_raw = deltas
            
            def scan_fn(gae, inp):
                delta, done = inp
                gae_prev = delta + self.gamma * self.gae_lambda * (1 - done) * gae
                return gae_prev, gae_prev
            
            _, gaes_prev = jax.lax.scan(scan_fn, gaes_raw[-1], (deltas[:-1], dones[:-1]), reverse=True)
            gaes_raw = jnp.concatenate([gaes_prev, gaes_raw[-1, None]], axis=0)  # (T,)
            
            # Targets = gaes + values (for critic update, matching InforMARL)
            targets = gaes_raw + values.squeeze(-1)  # (T,)
            
            # Broadcast to agent dimension (T, a)
            gaes_raw = gaes_raw[:, None].repeat(rewards.shape[1], axis=1)  # (T, a)
            
            # Return both targets and raw gaes (normalization will be done later, matching InforMARL)
            return targets, gaes_raw

        bT_targets, bTah_Aext = jax.vmap(compute_Aext_single)(
            rollout_rewards, bT_Ve, bTp1_Ve[:, 1:], rollout.dones
        )  # targets: (b, T), Aext: (b, T, a)

        # 5. Final advantages A_t^{(T)}
        def compute_A_final_single(Aext, Delta_team_int, z):
            return compute_conservative_exploration_gae(
                Tah_Aext=Aext,
                T_Delta_team_int=Delta_team_int,
                T_z=z  # z expected shape: (T, a)
            )
        bTah_A_final = jax.vmap(compute_A_final_single)(
            bTah_Aext, bT_Delta_team_int, rollout.zs.squeeze(-1)
        )  # (b, T, a)

        # 6. Normalize advantages
        bTah_A_final = (bTah_A_final - bTah_A_final.mean(axis=1, keepdims=True)) / (
            bTah_A_final.std(axis=1, keepdims=True) + 1e-8
        )
        
        # 7. Update networks
        def update_fn(carry, idx):
            critic, policy = carry
            rollout_batch = jtu.tree_map(lambda x: x[idx], rollout)
            # Use targets for critic update (matching InforMARL)
            critic, critic_info = self.update_critic(
                critic, rollout_batch, bT_targets[idx], rnn_states_Ve[idx], rnn_chunk_ids)
            policy, policy_info = self.update_policy(policy, rollout_batch, bTah_A_final[idx], rnn_chunk_ids)
            return (critic, policy), (critic_info | policy_info)

        (critic_train_state, policy_train_state), info = lax.scan(
            update_fn, (critic_train_state, policy_train_state), batch_idx
        )

        # get training info of the last PPO epoch
        info = jtu.tree_map(lambda x: x[-1], info)

        return critic_train_state, policy_train_state, info
    
    def update_critic(
            self,
            critic_train_state: TrainState,
            rollout: Rollout,
            targets: Array,
            rnn_states: Array,
            rnn_chunk_ids: Array
    ) -> Tuple[TrainState, dict]:
        # Create chunked rollout
        bcT_rollout = jax.tree.map(lambda x: x[:, rnn_chunk_ids], rollout)
        rnn_state_inits = jnp.zeros_like(rnn_states[:, rnn_chunk_ids[:, 0]])  # use zeros rnn_state as init

        def get_value_loss(params):
            values, value_rnn_states, final_value_rnn_states = jax.vmap(jax.vmap(
                ft.partial(self.scan_value,
                           critic_params=params)
            ))(bcT_rollout, rnn_state_inits)  # values: (b, n_chunks, T_chunk)
            # Align shapes: targets is (b, T, 1) -> squeeze last dim and flatten
            tgt = targets
            if tgt.ndim > 2:
                tgt = jnp.squeeze(tgt, axis=-1)  # (b, T)
            tgt = tgt.reshape((tgt.shape[0], -1))  # (b, *)

            values = values.reshape((values.shape[0], -1))  # (b, *)
            loss_critic = optax.l2_loss(values, tgt).mean()
            return loss_critic

        loss, grad = jax.value_and_grad(get_value_loss)(critic_train_state.params)
        critic_has_nan = has_any_nan_or_inf(grad).astype(jnp.float32)
        grad, grad_norm = compute_norm_and_clip(grad, self.max_grad_norm)
        critic_train_state = critic_train_state.apply_gradients(grads=grad)
        return critic_train_state, {'critic/loss': loss,
                                    'critic/grad_norm': grad_norm,
                                    'critic/has_nan': critic_has_nan,
                                    'critic/max_target': jnp.max(targets),
                                    'critic/min_target': jnp.min(targets)}

    def scan_eval_action(
            self,
            graphs: GraphsTuple,
            actions: Action,
            init_rnn_state: Array,
            action_keys: PRNGKey,
            actor_params: Params
    ) -> Tuple[Array, Array, Array, Array]:
        def body_(rnn_state, inp):
            graph, action, key = inp
            log_pi, entropy, new_rnn_state = self.policy.eval_action(actor_params, graph, action, rnn_state, key)
            return new_rnn_state, (log_pi, entropy, rnn_state)

        final_rnn_state, outputs = jax.lax.scan(body_, init_rnn_state, (graphs, actions, action_keys))
        log_pis, entropies, rnn_states = outputs

        return log_pis, entropies, rnn_states, final_rnn_state

    def update_policy(
            self, policy_train_state: TrainState, rollout: Rollout, gaes: Array, rnn_chunk_ids: Array
    ) -> Tuple[TrainState, dict]:
        # Final advantages are agent-wise (b, T, a)
        assert gaes.shape == (rollout.actions.shape[0], rollout.actions.shape[1], self.n_agents), \
            f"Expected gaes shape {(rollout.actions.shape[0], rollout.actions.shape[1], self.n_agents)}, got {gaes.shape}"

        # divide the rollout into chunks (n_env, n_chunks, T, ...)
        graph_chunks = jax.tree.map(lambda x: x[:, rnn_chunk_ids], rollout.graph)
        action_chunks = jax.tree.map(lambda x: x[:, rnn_chunk_ids], rollout.actions)
        rnn_state_inits = jnp.zeros_like(rollout.rnn_states[:, rnn_chunk_ids[:, 0]])  # use zeros rnn_state as init

        action_key = jr.fold_in(self.key, policy_train_state.step)
        action_keys = jr.split(action_key, rollout.actions.shape[0] * rollout.actions.shape[1]).reshape(
            rollout.actions.shape[:2] + (2,))
        action_keys = jax.tree.map(lambda x: x[:, rnn_chunk_ids], action_keys)

        def get_policy_loss(params):
            log_pis, policy_entropy, rnn_states, final_rnn_states = jax.vmap(jax.vmap(
                ft.partial(self.scan_eval_action,
                           actor_params=params)
            ))(graph_chunks, action_chunks, rnn_state_inits, action_keys)
            log_pis = log_pis.reshape((log_pis.shape[0], -1, log_pis.shape[-1]))

            ratio = jnp.exp(log_pis - rollout.log_pis)
            loss_policy1 = -ratio * gaes
            loss_policy2 = -jnp.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * gaes
            clip_frac = jnp.mean(loss_policy2 > loss_policy1)
            loss_policy = jnp.maximum(loss_policy1, loss_policy2).mean()
            total_entropy = policy_entropy.mean()
            policy_loss = loss_policy - self.coef_ent * total_entropy
            total_variation_dist = 0.5 * jnp.mean(jnp.abs(ratio - 1.0))
            return policy_loss, {'policy/clip_frac': clip_frac,
                                 'policy/entropy': policy_entropy.mean(),
                                 'policy/total_variation_dist': total_variation_dist}

        (loss, info), grad = jax.value_and_grad(get_policy_loss, has_aux=True)(policy_train_state.params)
        policy_has_nan = has_any_nan_or_inf(grad).astype(jnp.float32)

        # clip grad
        grad, grad_norm = compute_norm_and_clip(grad, self.max_grad_norm)

        # update policy
        policy_train_state = policy_train_state.apply_gradients(grads=grad)

        # get info
        info = {
                   'policy/loss': loss,
                   'policy/grad_norm': grad_norm,
                   'policy/has_nan': policy_has_nan,
                   'policy/log_pi_min': rollout.log_pis.min()
               } | info

        return policy_train_state, info

    def update_z(
        self,
        rollout: Rollout,
        z_old: Array  # Current budget (agent slots)
    ) -> Array:
        """
        Update budget z based on previous rollout's intrinsic reward sum
        
        z_new = sum of intrinsic rewards from previous rollout
        """
        # Compute total intrinsic reward per trajectory: sum over time and agents
        # rollout.intrinsic_rewards: (b, T, a)
        # Sum over time and agents to get total intrinsic return per trajectory
        intrinsic_return_per_traj = rollout.intrinsic_rewards.sum(axis=(1, 2))  # (b,)
        
        # Average over trajectories
        intrinsic_return_mean = intrinsic_return_per_traj.mean()
        
        # Set z to the mean intrinsic return (shared across all agents)
        z_new = jnp.full((self.n_agents, 1), intrinsic_return_mean)
        
        # Clip to valid range
        z_new = jnp.clip(z_new, -self._env.reward_max, -self._env.reward_min)
        
        # EMA approach (commented out - design choice)
        # lr_z = 1e-3  # or self.lr_z
        # z_update = lr_z * intrinsic_return_mean
        # z_new = z_old + z_update
        # z_new = jnp.clip(z_new, -self._env.reward_max, -self._env.reward_min)
        
        return z_new

    def save(self, save_dir: str, step: int):
        model_dir = os.path.join(save_dir, str(step))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        pickle.dump(self.policy_train_state.params, open(os.path.join(model_dir, 'actor.pkl'), 'wb'))
        pickle.dump(self.critic_train_state.params, open(os.path.join(model_dir, 'critic.pkl'), 'wb'))

    def get_opt_z(
        self,
        graph: GraphsTuple,
        rnn_state: Array,
        params: Optional[Params] = None
    ) -> Tuple[Array, Array]:
        """
        Get z for evaluation. For EFXplorer, we use z=0 for evaluation.
        
        Returns
        -------
        z: Array
            Budget z set to 0 for evaluation (shape: (n_agents, 1))
        rnn_state: Array
            Unchanged RNN state
        """
        # For evaluation, use z=0 (initial budget)
        z = jnp.zeros((self.n_agents, 1))
        return z, rnn_state

    def load(self, load_dir: str, step: int):
        path = os.path.join(load_dir, str(step))

        self.policy_train_state = \
            self.policy_train_state.replace(params=pickle.load(open(os.path.join(path, 'actor.pkl'), 'rb')))
        self.critic_train_state = \
            self.critic_train_state.replace(params=pickle.load(open(os.path.join(path, 'critic.pkl'), 'rb')))

    