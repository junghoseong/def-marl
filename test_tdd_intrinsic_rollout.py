"""
TDD Intrinsic Reward 검증 스크립트

무작위 rollout을 통해 TDD intrinsic reward가 제대로 계산되는지 검증합니다.

Usage:
    python test_tdd_intrinsic_rollout.py --env MPETarget -n 3 --obs 3
"""

import argparse
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from defmarl.env import make_env
from defmarl.algo import make_algo
from defmarl.algo.module.tdd_intrinsic import (
    create_tdd_train_state,
    tdd_train_step,
    tdd_intrinsic_reward,
    tdd_loss
)


def extract_node_features(rollout, agent_id: int):
    """
    Rollout에서 특정 에이전트의 node features 추출
    
    Args:
        rollout: Rollout 객체
        agent_id: 에이전트 ID
    
    Returns:
        obs_t: (T-1, n_envs, node_dim) - 현재 상태
        obs_tp1: (T-1, n_envs, node_dim) - 다음 상태
    """
    graphs = rollout.graph
    
    # Rollout 구조: jax.vmap으로 여러 환경에 대해 병렬 실행
    # rollout_efxplorer는 jax.lax.scan을 사용하므로 (T, ...) 구조 반환
    # vmap 후에는 (n_envs, T, ...) 구조가 됨
    
    # nodes shape: (n_envs, T, n_agents, node_dim)
    # rewards shape를 참고하여 구조 확인
    rewards_shape = rollout.rewards.shape  # (n_envs, T, n_agents) 또는 (T, n_envs, n_agents)
    
    if graphs.nodes.ndim == 3:
        # (T, n_agents, node_dim) - single env (테스트용)
        obs_t = graphs.nodes[:-1, agent_id, :]  # (T-1, node_dim)
        obs_tp1 = graphs.nodes[1:, agent_id, :]  # (T-1, node_dim)
        # Add env dimension
        obs_t = obs_t[:, None, :]  # (T-1, 1, node_dim)
        obs_tp1 = obs_tp1[:, None, :]  # (T-1, 1, node_dim)
    elif graphs.nodes.ndim == 4:
        # rewards shape로 구조 판단
        if len(rewards_shape) == 3 and rewards_shape[0] < rewards_shape[1]:
            # (n_envs, T, n_agents, node_dim) - 일반적인 경우
            obs_t = graphs.nodes[:, :-1, agent_id, :]  # (n_envs, T-1, node_dim)
            obs_tp1 = graphs.nodes[:, 1:, agent_id, :]  # (n_envs, T-1, node_dim)
            # Transpose to (T-1, n_envs, node_dim)
            obs_t = jnp.transpose(obs_t, (1, 0, 2))
            obs_tp1 = jnp.transpose(obs_tp1, (1, 0, 2))
        else:
            # (T, n_envs, n_agents, node_dim) - 다른 구조
            obs_t = graphs.nodes[:-1, :, agent_id, :]  # (T-1, n_envs, node_dim)
            obs_tp1 = graphs.nodes[1:, :, agent_id, :]  # (T-1, n_envs, node_dim)
    else:
        raise ValueError(f"Unexpected graphs.nodes.ndim: {graphs.nodes.ndim}, shape: {graphs.nodes.shape}")
    
    return obs_t, obs_tp1


def compute_tdd_intrinsic_from_rollout(rollout, tdd_states, node_dim: int):
    """
    Rollout에서 TDD intrinsic reward 계산
    
    Args:
        rollout: Rollout 객체
        tdd_states: List of TDD train states (각 에이전트별)
        node_dim: Node feature dimension
    
    Returns:
        intrinsic_rewards: (T, n_envs, n_agents) intrinsic rewards
    """
    n_agents = len(tdd_states)
    
    # Rollout shape 확인
    T = rollout.rewards.shape[0]  # time steps
    n_envs = rollout.rewards.shape[1]  # parallel environments
    
    intrinsic_rewards_list = []
    
    for agent_id in range(n_agents):
        # Node features 추출
        obs_t, obs_tp1 = extract_node_features(rollout, agent_id)
        
        # Shape: (T-1, n_envs, node_dim) -> (T-1 * n_envs, node_dim)
        obs_t_flat = obs_t.reshape(-1, node_dim)
        obs_tp1_flat = obs_tp1.reshape(-1, node_dim)
        
        # TDD intrinsic reward 계산
        tdd_state = tdd_states[agent_id]
        intrinsic_flat = tdd_intrinsic_reward(
            tdd_state.params,
            tdd_state.apply_fn_enc,
            obs_t_flat,
            obs_tp1_flat,
            aggregate="min"
        )  # (T-1 * n_envs,)
        
        # Reshape back to (T-1, n_envs)
        intrinsic_reshaped = intrinsic_flat.reshape(T - 1, n_envs)
        
        # 마지막 timestep은 0으로 패딩
        intrinsic_padded = jnp.concatenate([
            intrinsic_reshaped,
            jnp.zeros((1, n_envs))
        ], axis=0)  # (T, n_envs)
        
        intrinsic_rewards_list.append(intrinsic_padded)
    
    # Stack: (T, n_envs, n_agents)
    intrinsic_rewards = jnp.stack(intrinsic_rewards_list, axis=-1)
    
    return intrinsic_rewards


def test_tdd_intrinsic(args):
    """TDD intrinsic reward 검증 테스트"""
    print("=" * 80)
    print("TDD Intrinsic Reward 검증 테스트")
    print("=" * 80)
    
    # Seed 설정
    rng = jax.random.PRNGKey(args.seed)
    
    # 환경 생성
    print(f"\n환경 생성: {args.env}, Agents: {args.num_agents}, Obs: {args.obs}")
    env = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        full_observation=args.full_observation,
        area_size=args.area_size,
    )
    
    print(f"Node dim: {env.node_dim}, State dim: {env.state_dim}, Action dim: {env.action_dim}")
    
    # 알고리즘 생성 (무작위 정책으로 rollout 수집)
    print(f"\n알고리즘 생성: {args.algo}")
    algo = make_algo(
        algo=args.algo,
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=env.num_agents,
        cost_weight=args.cost_weight,
        actor_gnn_layers=args.gnn_layers,
        critic_gnn_layers=args.gnn_layers,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        seed=args.seed,
        batch_size=args.batch_size,
        use_rnn=not args.no_rnn,
        use_lstm=args.use_lstm,
        rnn_step=args.rnn_step,
        gamma=0.99,
        clip_eps=args.clip_eps,
    )
    
    # TDD 네트워크 초기화
    print(f"\nTDD 네트워크 초기화...")
    tdd_config = {
        'hidden_dim': args.tdd_hidden_dim,
        'tdd_lr': args.tdd_lr,
        'energy_fn': args.tdd_energy_fn,
        'loss_fn': args.tdd_loss_fn,
    }
    
    rng, *tdd_rngs = jax.random.split(rng, env.num_agents + 1)
    tdd_states = [
        create_tdd_train_state(
            tdd_rngs[i],
            obs_dim=env.node_dim,
            hidden_dim=tdd_config['hidden_dim'],
            lr=tdd_config['tdd_lr'],
            energy_fn=tdd_config['energy_fn'],
            loss_fn=tdd_config['loss_fn']
        )
        for i in range(env.num_agents)
    ]
    print(f"TDD 네트워크 생성 완료: {len(tdd_states)}개 (에이전트별)")
    
    # Rollout 수집
    print(f"\n무작위 rollout 수집 중...")
    n_envs = args.n_env_test
    rng, rollout_rng = jax.random.split(rng)
    rollout_keys = jax.random.split(rollout_rng, n_envs)
    
    rollout = algo.collect(algo.params, rollout_keys)
    
    print(f"Rollout shape:")
    print(f"  rewards: {rollout.rewards.shape}")
    print(f"  intrinsic_rewards: {rollout.intrinsic_rewards.shape}")
    print(f"  graph.nodes: {rollout.graph.nodes.shape}")
    print(f"  next_graph.nodes: {rollout.next_graph.nodes.shape}")
    
    # Rollout 구조 확인
    print(f"\nRollout 구조 확인:")
    print(f"  rewards shape: {rollout.rewards.shape}")
    print(f"  graph.nodes shape: {rollout.graph.nodes.shape}")
    print(f"  graph.nodes ndim: {rollout.graph.nodes.ndim}")
    
    # TDD intrinsic reward 계산
    print(f"\nTDD intrinsic reward 계산 중...")
    try:
        tdd_intrinsic = compute_tdd_intrinsic_from_rollout(
            rollout, tdd_states, env.node_dim
        )
        
        print(f"TDD intrinsic reward shape: {tdd_intrinsic.shape}")
        print(f"TDD intrinsic reward 통계:")
        print(f"  Mean: {float(jnp.mean(tdd_intrinsic)):.6f}")
        print(f"  Std: {float(jnp.std(tdd_intrinsic)):.6f}")
        print(f"  Min: {float(jnp.min(tdd_intrinsic)):.6f}")
        print(f"  Max: {float(jnp.max(tdd_intrinsic)):.6f}")
    except Exception as e:
        print(f"  ✗ TDD intrinsic reward 계산 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 에이전트별 통계
    print(f"\n에이전트별 TDD intrinsic reward 통계:")
    for agent_id in range(env.num_agents):
        agent_intrinsic = tdd_intrinsic[:, :, agent_id]
        print(f"  Agent {agent_id}: mean={float(jnp.mean(agent_intrinsic)):.6f}, "
              f"std={float(jnp.std(agent_intrinsic)):.6f}, "
              f"min={float(jnp.min(agent_intrinsic)):.6f}, "
              f"max={float(jnp.max(agent_intrinsic)):.6f}")
    
    # TDD 네트워크 학습 테스트
    print(f"\nTDD 네트워크 학습 테스트...")
    for agent_id in range(env.num_agents):
        obs_t, obs_tp1 = extract_node_features(rollout, agent_id)
        
        # Flatten for training
        obs_t_flat = obs_t.reshape(-1, env.node_dim)
        obs_tp1_flat = obs_tp1.reshape(-1, env.node_dim)
        
        # 학습
        tdd_state = tdd_states[agent_id]
        tdd_state_new, tdd_metrics = tdd_train_step(
            tdd_state,
            obs_t_flat,
            obs_tp1_flat,
            energy_fn=tdd_config['energy_fn'],
            loss_fn=tdd_config['loss_fn']
        )
        
        print(f"  Agent {agent_id}: loss={float(tdd_metrics['tdd/loss']):.6f}, "
              f"acc={float(tdd_metrics['tdd/acc']):.4f}")
        
        tdd_states[agent_id] = tdd_state_new
    
    # 학습 후 intrinsic reward 재계산
    print(f"\n학습 후 TDD intrinsic reward 재계산...")
    tdd_intrinsic_after = compute_tdd_intrinsic_from_rollout(
        rollout, tdd_states, env.node_dim
    )
    
    print(f"학습 후 TDD intrinsic reward 통계:")
    print(f"  Mean: {float(jnp.mean(tdd_intrinsic_after)):.6f}")
    print(f"  Std: {float(jnp.std(tdd_intrinsic_after)):.6f}")
    
    # 변화량 확인
    intrinsic_diff = tdd_intrinsic_after - tdd_intrinsic
    print(f"\nIntrinsic reward 변화량:")
    print(f"  Mean diff: {float(jnp.mean(intrinsic_diff)):.6f}")
    print(f"  Std diff: {float(jnp.std(intrinsic_diff)):.6f}")
    
    # 검증: NaN/Inf 체크
    print(f"\n검증:")
    has_nan = jnp.any(jnp.isnan(tdd_intrinsic))
    has_inf = jnp.any(jnp.isinf(tdd_intrinsic))
    print(f"  NaN 체크: {bool(has_nan)}")
    print(f"  Inf 체크: {bool(has_inf)}")
    
    if not has_nan and not has_inf:
        print("  ✓ TDD intrinsic reward 계산 성공!")
    else:
        print("  ✗ TDD intrinsic reward 계산 실패 (NaN/Inf 발견)")
    
    print("\n" + "=" * 80)
    print("테스트 완료!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    
    # Required arguments
    parser.add_argument("--env", type=str, default="MPETarget")
    parser.add_argument("--algo", type=str, default="efxplorer")
    parser.add_argument("-n", "--num-agents", type=int, default=3)
    parser.add_argument("--obs", type=int, default=3)
    
    # Environment arguments
    parser.add_argument('--full-observation', action='store_true', default=False)
    parser.add_argument("--area-size", type=float, default=None)
    
    # Algorithm arguments
    parser.add_argument("--cost-weight", type=float, default=0.)
    parser.add_argument("--clip-eps", type=float, default=0.25)
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-critic", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--no-rnn", action="store_true", default=False)
    parser.add_argument("--use-lstm", action="store_true", default=False)
    parser.add_argument("--rnn-step", type=int, default=15)
    
    # Test arguments
    parser.add_argument("--n-env-test", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    
    # TDD arguments
    parser.add_argument("--tdd-hidden-dim", type=int, default=256)
    parser.add_argument("--tdd-lr", type=float, default=3e-4)
    parser.add_argument("--tdd-energy-fn", type=str, default="mrn_pot")
    parser.add_argument("--tdd-loss-fn", type=str, default="infonce")
    
    args = parser.parse_args()
    test_tdd_intrinsic(args)


if __name__ == "__main__":
    main()

