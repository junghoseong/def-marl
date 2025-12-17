# TDD Intrinsic Reward 통합 가이드

## 개요
`test_successor_distance.py`의 Successor Distance (SD) intrinsic reward를 `train.py`의 학습 환경에 통합하는 방법입니다.

## 현재 상황 분석

### `test_successor_distance.py`
- **목적**: SD 네트워크 학습 검증용 독립 테스트 코드
- **환경**: 합성 궤적 데이터 (simple_corridor_dilemma 패턴)
- **기능**: 
  - SD 네트워크 학습 (`train_sd()`)
  - SD heatmap 시각화
  - 독립 실행 가능

### `train.py` 구조
- **프레임워크**: `def-marl` (Trainer 기반)
- **학습 루프**: `Trainer.train()` → `algo.collect()` → `algo.update()`
- **기존 모듈**: `defmarl/algo/module/tdd_intrinsic.py` 존재

## 통합 단계

### 1단계: 알고리즘 클래스에 TDD 추가

#### 1.1 TDD 네트워크 초기화
```python
# defmarl/algo/your_algo.py (예: informarl.py, efxplorer.py 등)

from defmarl.algo.module.tdd_intrinsic import (
    create_tdd_train_state, 
    tdd_train_step,
    tdd_intrinsic_reward
)

class YourAlgorithm(Algorithm):
    def __init__(self, ..., use_tdd: bool = False, tdd_config: dict = None):
        # ... 기존 초기화 코드 ...
        
        # TDD 초기화
        self.use_tdd = use_tdd
        if use_tdd:
            tdd_config = tdd_config or {}
            obs_dim = node_dim  # 또는 적절한 observation dimension
            
            # 각 에이전트별 TDD 네트워크 생성
            rng, *tdd_rngs = jax.random.split(rng, self.n_agents + 1)
            self.tdd_states = [
                create_tdd_train_state(
                    tdd_rngs[i],
                    obs_dim=obs_dim,
                    hidden_dim=tdd_config.get('hidden_dim', 256),
                    lr=tdd_config.get('tdd_lr', 3e-4),
                    energy_fn=tdd_config.get('energy_fn', 'mrn_pot'),
                    loss_fn=tdd_config.get('loss_fn', 'infonce')
                )
                for i in range(self.n_agents)
            ]
```

#### 1.2 Rollout에서 Intrinsic Reward 계산
```python
def collect(self, params: Params, b_key: PRNGKey) -> Rollout:
    # 기존 rollout 수집
    rollout = self.rollout_fn(params, b_key)
    
    if self.use_tdd:
        # Intrinsic reward 계산
        intrinsic_rewards = self._compute_tdd_intrinsic(rollout)
        
        # Rollout 업데이트
        rollout = rollout._replace(intrinsic_rewards=intrinsic_rewards)
    
    return rollout

def _compute_tdd_intrinsic(self, rollout: Rollout) -> jnp.ndarray:
    """
    rollout.graphs: (T, n_envs, ...) - GraphsTuple
    Returns: (T, n_envs, n_agents) intrinsic rewards
    """
    T, n_envs = rollout.graphs.nodes.shape[:2]
    intrinsic_rewards = []
    
    for agent_id in range(self.n_agents):
        agent_intrinsic = []
        tdd_state = self.tdd_states[agent_id]
        
        for t in range(T - 1):
            # 현재/다음 상태 추출 (node features에서)
            obs_t = rollout.graphs.nodes[t, :, agent_id]  # (n_envs, node_dim)
            obs_tp1 = rollout.graphs.nodes[t+1, :, agent_id]  # (n_envs, node_dim)
            
            # Intrinsic reward 계산
            intrinsic = tdd_intrinsic_reward(
                tdd_state.params,
                tdd_state.apply_fn_enc,
                obs_t,
                obs_tp1,
                aggregate="min"  # 또는 config에서 가져오기
            )  # (n_envs,)
            
            agent_intrinsic.append(intrinsic)
        
        # 마지막 timestep은 0으로 패딩
        agent_intrinsic.append(jnp.zeros(n_envs))
        agent_intrinsic = jnp.stack(agent_intrinsic, axis=0)  # (T, n_envs)
        intrinsic_rewards.append(agent_intrinsic)
    
    # Stack: (T, n_envs, n_agents)
    return jnp.stack(intrinsic_rewards, axis=-1)
```

#### 1.3 Update에서 TDD 학습 및 Reward 결합
```python
def update(self, rollout: Rollout, step: int) -> dict:
    update_info = {}
    
    # TDD 네트워크 학습
    if self.use_tdd:
        for agent_id in range(self.n_agents):
            tdd_state = self.tdd_states[agent_id]
            
            # 배치 구성: (obs_t, obs_tp1) 쌍
            T, n_envs = rollout.graphs.nodes.shape[:2]
            obs_t = rollout.graphs.nodes[:-1, :, agent_id].reshape(-1, self.node_dim)
            obs_tp1 = rollout.graphs.nodes[1:, :, agent_id].reshape(-1, self.node_dim)
            
            # TDD 학습
            tdd_state, tdd_metrics = tdd_train_step(
                tdd_state,
                obs_t,
                obs_tp1,
                energy_fn=self.tdd_config.get('energy_fn', 'mrn_pot'),
                loss_fn=self.tdd_config.get('loss_fn', 'infonce')
            )
            self.tdd_states[agent_id] = tdd_state
            update_info[f'tdd/agent_{agent_id}/loss'] = float(tdd_metrics['tdd/loss'])
    
    # Reward 결합
    if self.use_tdd:
        intrinsic_lambda = self.tdd_config.get('intrinsic_lambda', 0.1)
        combined_rewards = rollout.rewards + intrinsic_lambda * rollout.intrinsic_rewards.sum(axis=-1)
        rollout = rollout._replace(rewards=combined_rewards)
    
    # 기존 PPO 업데이트
    # ... 기존 update 코드 ...
    
    return update_info
```

### 2단계: train.py에 TDD 옵션 추가

```python
# train.py 수정

def train(args):
    # ... 기존 코드 ...
    
    # TDD 설정
    use_tdd = getattr(args, 'use_tdd', False)
    tdd_config = {
        'hidden_dim': getattr(args, 'tdd_hidden_dim', 256),
        'tdd_lr': getattr(args, 'tdd_lr', 3e-4),
        'energy_fn': getattr(args, 'tdd_energy_fn', 'mrn_pot'),
        'loss_fn': getattr(args, 'tdd_loss_fn', 'infonce'),
        'intrinsic_lambda': getattr(args, 'tdd_intrinsic_lambda', 0.1),
    }
    
    # 알고리즘 생성 시 TDD 옵션 전달
    algo = make_algo(
        algo=args.algo,
        env=env,
        # ... 기존 인자들 ...
        use_tdd=use_tdd,
        tdd_config=tdd_config,
    )
    
    # ... 나머지 코드 ...
```

### 3단계: make_algo 함수 수정

```python
# defmarl/algo/__init__.py 또는 해당 파일

def make_algo(..., use_tdd: bool = False, tdd_config: dict = None):
    if algo == "informarl":
        return InforMARL(
            # ... 기존 인자들 ...
            use_tdd=use_tdd,
            tdd_config=tdd_config,
        )
    elif algo == "efxplorer":
        return EFXplorer(
            # ... 기존 인자들 ...
            use_tdd=use_tdd,
            tdd_config=tdd_config,
        )
    # ...
```

## 사용 예시

```bash
# TDD 없이 학습
python train.py --env YourEnv --algo informarl -n 5 --obs 10

# TDD 포함 학습
python train.py --env YourEnv --algo informarl -n 5 --obs 10 \
    --use-tdd \
    --tdd-hidden-dim 256 \
    --tdd-lr 3e-4 \
    --tdd-intrinsic-lambda 0.1
```

## 주의사항

1. **Observation Dimension**: 
   - `test_successor_distance.py`는 2D position (또는 4D with goal)
   - 실제 환경의 observation dimension에 맞춰 조정 필요

2. **Feature Extraction**:
   - Graph-based 환경의 경우 `rollout.graphs.nodes`에서 적절한 feature 추출 필요
   - 필요시 encoder 추가 고려

3. **Hyperparameters**:
   - `intrinsic_lambda`: extrinsic과 intrinsic reward의 가중치
   - TDD 학습률과 업데이트 빈도 조정 필요

4. **성능**:
   - TDD 네트워크 학습은 추가 계산 비용 발생
   - 필요시 업데이트 빈도 조정 (매 step마다 학습하지 않아도 됨)

## 참고 자료

- `defmarl/algo/module/tdd_intrinsic.py`: TDD 모듈 구현
- `JaxMARL_larr/baselines/MAPPO/mappo_rnn_corridor.py`: TDD 통합 예시 (다른 프레임워크)
- `test_successor_distance.py`: SD 학습 검증 코드

