# DefMARL 클래스 구조 분석

이 문서는 `def-marl/defmarl/algo/defmarl.py`의 `DefMARL` 클래스 구조를 상세히 분석한 것입니다.

## 개요

**핵심 알고리즘**: **Dec-EFOCP (Decentralized Epigraph Form Optimal Control Problem)** 기반의 Multi-Agent Constrained RL

**상속 구조**: `DefMARL` → `Algorithm` (base class)

---

## 클래스 초기화 (`__init__`)

### 입력 파라미터

#### 환경 관련
- `env: MultiAgentEnv`: Multi-agent 환경 객체
- `node_dim: int`: 그래프 노드 feature 차원
- `edge_dim: int`: 그래프 엣지 feature 차원
- `state_dim: int`: 상태 차원
- `action_dim: int`: 액션 차원
- `n_agents: int`: 에이전트 수

#### 하이퍼파라미터
- `cost_weight: float = 0.`: Cost 가중치 (사용되지 않음)
- `actor_gnn_layers: int = 2`: Actor 네트워크의 GNN 레이어 수
- `critic_gnn_layers: int = 2`: Critic 네트워크의 GNN 레이어 수
- `Vh_gnn_layers: int = 1`: Constraint value 네트워크의 GNN 레이어 수
- `gamma: float = 0.99`: 할인 인자
- `lr_actor: float = 1e-5`: Actor 학습률
- `lr_critic: float = 1e-5`: Critic 학습률
- `batch_size: int = 8192`: 배치 크기
- `epoch_ppo: int = 1`: PPO 업데이트 에폭 수
- `clip_eps: float = 0.25`: PPO clipping epsilon
- `gae_lambda: float = 0.95`: GAE lambda 파라미터
- `coef_ent: float = 1e-2`: 엔트로피 계수
- `max_grad_norm: float = 2.0`: 최대 gradient norm (clipping)
- `seed: int = 0`: 랜덤 시드

#### RNN 관련
- `use_rnn: bool = True`: RNN 사용 여부
- `rnn_layers: int = 1`: RNN 레이어 수
- `rnn_step: int = 16`: RNN chunk 크기
- `use_lstm: bool = False`: LSTM 사용 여부 (False면 GRU)

#### 기타
- `rollout_length: Optional[int] = None`: Rollout 길이 (None이면 env 기본값)
- `coef_ent_schedule: bool = False`: 엔트로피 계수 스케줄링 여부
- `use_prev_init: bool = False`: 이전 rollout의 마지막 상태를 초기 상태로 사용 여부

### 초기화 과정

1. **Base 클래스 초기화**: `Algorithm.__init__()` 호출
2. **하이퍼파라미터 저장**: 모든 파라미터를 인스턴스 변수로 저장
3. **z 범위 설정**: 
   - `self.z_min = -env.reward_max`
   - `self.z_max = -env.reward_min`
4. **Nominal graph 생성**: 네트워크 초기화를 위한 더미 그래프
5. **네트워크 초기화**:
   - Policy (Actor) 네트워크
   - Critic (Value) 네트워크
   - Vh (Constraint Value) 네트워크
6. **Root Finder 초기화**: z 최적화를 위한 root finder
7. **Rollout 함수 설정**: JIT 컴파일된 rollout 함수

---

## 주요 인스턴스 변수

### 네트워크 관련
- `self.policy: PPOPolicy`: Actor 정책 네트워크
- `self.critic: ValueNet`: Value 함수 네트워크 (Vl)
- `self.Vh: ValueNet`: Constraint value 함수 네트워크
- `self.policy_train_state: TrainState`: Policy 학습 상태
- `self.critic_train_state: TrainState`: Critic 학습 상태
- `self.Vh_train_state: TrainState`: Vh 학습 상태

### RNN 상태
- `self.init_rnn_state: Array`: Policy RNN 초기 상태 `(rnn_layers, n_agents, n_carries, rnn_state_dim)`
- `self.init_Vl_rnn_state: Array`: Critic RNN 초기 상태 `(rnn_layers, 1, n_carries, rnn_state_dim)`
- `self.init_Vh_rnn_state: Array`: Vh RNN 초기 상태 `(rnn_layers, n_agents, n_carries, rnn_state_dim)`

### 기타
- `self.nominal_graph: GraphsTuple`: 네트워크 초기화용 더미 그래프
- `self.nominal_z: Array`: 초기 z 값 `(n_agents, 1)`
- `self.root_finder: RootFinder`: z 최적화를 위한 root finder
- `self.rollout_fn`: JIT 컴파일된 rollout 함수
- `self.get_init_graph`: 초기 그래프 생성 함수
- `self.memory: Rollout`: 이전 rollout 메모리 (use_prev_init=True일 때 사용)
- `self.key: PRNGKey`: JAX 랜덤 키

---

## 기타 메서드 (요약)

### Properties
- **`config`**: 하이퍼파라미터 딕셔너리 반환
- **`params`**: 네트워크 파라미터 딕셔너리 반환

### Action & Value Methods
- **`get_opt_z(graph, Vh_rnn_state, params=None)`**: RootFinder를 사용하여 최적 z 값 계산
- **`act(graph, z, rnn_state, params=None)`**: 정책에서 액션 샘플링 (평가/추론용)
- **`get_value(graph, z, rnn_state, params=None)`**: Value 함수 값 계산
- **`step(graph, z, rnn_state, key, params=None)`**: 학습용 액션 샘플링 (log probability 포함)

### Data Collection
- **`collect(params, b_key)`**: 여러 환경에서 병렬로 rollout 수집
  - `use_prev_init=True`이면 이전 rollout의 마지막 상태를 초기 상태로 사용

### Training Entry Point
- **`update(rollout, step)`**: 학습 메인 함수
  - 배치 인덱스 셔플 및 RNN chunk 분할
  - `epoch_ppo` 횟수만큼 `update_inner` 반복 호출
  - 자세한 내용은 아래 `update_inner` 섹션 참조

### Utility Methods
- **`save(save_dir, step)`**: Policy, Critic, Vh 파라미터를 pickle로 저장
- **`load(load_dir, step)`**: 저장된 파라미터를 로드하여 train_state 업데이트

---


## 핵심 Training 메서드 상세 분석

### 1. `update_inner()` - 전체 학습 프로세스의 핵심

**위치**: `defmarl.py:413-481`

**역할**: 수집된 rollout 데이터로 Policy, Critic, Vh 네트워크를 업데이트하는 메인 학습 함수 (JIT 컴파일됨)

**입력**:
- `critic_train_state, Vh_train_state, policy_train_state`: 학습 상태
- `rollout: Rollout`: 수집된 rollout 데이터 `(n_env, T, n_agent, ...)`
- `batch_idx: Array`: 배치 인덱스 (셔플된 환경 인덱스)
- `rnn_chunk_ids: Array`: RNN chunk 인덱스 (긴 시퀀스를 chunk로 분할)

**처리 단계**:

#### Step 1: Value 함수 계산 (`scan_value`)
```python
(bT_Vl, bTah_Vh), (rnn_states_Vl, rnn_states_Vh), (final_rnn_states_Vl, final_rnn_states_Vh) = (
    jax_vmap(scan_value)(rollout))
```
- 모든 타임스텝에 대해 Critic (Vl)과 Vh 값을 계산
- `bT_Vl`: `(batch, T)` - 각 타임스텝의 reward value
- `bTah_Vh`: `(batch, T, n_agents, n_cost)` - 각 타임스텝, 각 에이전트, 각 constraint의 cost value

#### Step 2: Final Value 계산
```python
final_Vl, final_Vh = jax_vmap(final_value_fn)(
    rollout.next_graph, rollout.zs, final_rnn_states_Vl, final_rnn_states_Vh)
bTp1_Vl = jnp.concatenate([bT_Vl, final_Vl[:, None]], axis=1)
bTp1ah_Vh = jnp.concatenate([bTah_Vh, final_Vh[:, None]], axis=1)
```
- 마지막 상태의 value를 계산하여 `T+1` 타임스텝까지 확장
- GAE 계산을 위해 필요

#### Step 3: Dec-EFOCP GAE 계산 (`compute_dec_efocp_gae`)
```python
bTah_Qh, bT_Ql, bTa_Q = jax.vmap(
    ft.partial(compute_dec_efocp_gae, disc_gamma=self.gamma, gae_lambda=self.gae_lambda)
)(Tah_hs=rollout.costs,
  T_l=-rollout.rewards,
  T_z=rollout.zs.squeeze(-1)[:, :, 0],
  Tp1ah_Vh=bTp1ah_Vh,
  Tp1_Vl=bTp1_Vl)
```
- **핵심**: Dec-EFOCP 전용 GAE 계산 (자세한 설명은 아래 `compute_dec_efocp_gae` 섹션 참조)
- `bTah_Qh`: `(batch, T, n_agents, n_cost)` - Constraint Q 함수
- `bT_Ql`: `(batch, T)` - Reward Q 함수
- `bTa_Q`: `(batch, T, n_agents)` - Combined Q 함수

#### Step 4: Advantage 계산 및 정규화
```python
bTa_V = jax_vmap(jax_vmap(compute_dec_efocp_V))(
    rollout.zs.squeeze(-1)[:, :, 0], bTah_Vh, bT_Vl)
bTa_A = bTa_Q - bTa_V
bTa_A = (bTa_A - bTa_A.mean(axis=1, keepdims=True)) / (bTa_A.std(axis=1, keepdims=True) + 1e-8)
```
- `compute_dec_efocp_V`로 현재 value 계산
- Advantage = Q - V
- **정규화**: 각 환경 내에서 advantage를 정규화 (mean=0, std=1)

#### Step 5: 배치별 네트워크 업데이트
```python
def update_fn(carry, idx):
    critic, Vh, policy = carry
    rollout_batch = jtu.tree_map(lambda x: x[idx], rollout)
    critic, Vh, value_info = self.update_value(...)
    policy, policy_info = self.update_policy(...)
    return (critic, Vh, policy), (value_info | policy_info)

(critic_train_state, Vh_train_state, policy_train_state), info = lax.scan(
    update_fn, (critic_train_state, Vh_train_state, policy_train_state), batch_idx
)
```
- 배치를 순차적으로 처리하며 네트워크 업데이트
- `update_value`: Critic과 Vh 업데이트
- `update_policy`: Policy 업데이트 (PPO)

**반환값**: 업데이트된 학습 상태와 학습 정보 딕셔너리

---

### 2. `scan_value()` - Value 함수 순차 계산

**위치**: `defmarl.py:381-411`

**역할**: Rollout 전체에 대해 Critic (Vl)과 Vh 네트워크를 순차적으로 forward pass하여 value 값 계산

**입력**:
- `rollout: Rollout`: Rollout 데이터
- `init_rnn_state_Vl, init_rnn_state_Vh`: RNN 초기 상태
- `critic_params, Vh_params`: 네트워크 파라미터

**동작**:
```python
def body_(rnn_state, inp):
    graph, z = inp
    rnn_state_Vl, rnn_state_Vh = rnn_state
    # Critic: z[0]만 사용 (모든 에이전트가 같은 z 사용)
    value, new_rnn_state_V = self.critic.get_value(
        critic_params, graph, rnn_state_Vl, z[0][None, :])
    # Vh: 각 에이전트별 z 사용
    value_h, new_rnn_state_Vh = self.Vh.get_value(
        Vh_params, graph, rnn_state_Vh, z)
    return (new_rnn_state_V, new_rnn_state_Vh), (value, value_h, rnn_state_Vl, rnn_state_Vh)
```

**핵심 포인트**:
- **Critic (Vl)**: 모든 에이전트가 동일한 z 값 사용 (`z[0]`)
- **Vh**: 각 에이전트가 자신의 z 값 사용 (`z` shape: `(n_agents, 1)`)
- RNN 상태를 순차적으로 업데이트하며 모든 타임스텝의 value 계산

**반환값**:
- `(T_Vl, Tah_Vh)`: Value 값들
  - `T_Vl`: `(T,)` - 각 타임스텝의 reward value
  - `Tah_Vh`: `(T, n_agents, n_cost)` - 각 타임스텝, 각 에이전트, 각 constraint의 cost value
- `(rnn_states_Vl, rnn_states_Vh)`: 모든 타임스텝의 RNN 상태
- `(final_rnn_state_Vl, final_rnn_state_Vh)`: 최종 RNN 상태 (final value 계산용)

---

### 3. `compute_dec_efocp_gae()` - Dec-EFOCP GAE 계산

**위치**: `utils.py:40-115`

**역할**: Dec-EFOCP를 위한 Generalized Advantage Estimation 계산. **Backward Dynamic Programming**을 사용하여 역순으로 계산.

**입력**:
- `Tah_hs: Array`: `(T, n_agents, n_cost)` - 각 타임스텝의 cost
- `T_l: Array`: `(T,)` - 각 타임스텝의 negative reward (`-reward`)
- `T_z: Array`: `(T, n_agents)` - 각 타임스텝, 각 에이전트의 z 값
- `Tp1ah_Vh: Array`: `(T+1, n_agents, n_cost)` - 다음 상태의 Vh 값
- `Tp1_Vl: Array`: `(T+1,)` - 다음 상태의 Vl 값
- `disc_gamma: float`: 할인 인자
- `gae_lambda: float`: GAE lambda 파라미터
- `discount_to_max: bool = True`: Vh 계산 시 max 사용 여부

**핵심 알고리즘**:

#### Backward DP Loop
```python
def loop(carry, inp):
    ii, hs, l, z, Vhs, Vl = inp  # 현재 타임스텝 정보
    next_Vhs_row, next_Vl_row, gae_coeffs = carry  # 이전 계산 결과
    
    # Step 1: Vh DP 업데이트
    if discount_to_max:
        h_disc = hs.max(-1)  # 각 에이전트의 최대 cost
    else:
        h_disc = hs
    
    # Vh = max(현재 cost, γ·다음 Vh)
    disc_to_h = (1 - disc_gamma) * h_disc[None, :, None] + disc_gamma * next_Vhs_row
    Vhs_row = jnp.maximum(hs, disc_to_h)
    
    # Step 2: Vl DP 업데이트
    # Vl = 현재 reward + γ·다음 Vl
    Vl_row = l + disc_gamma * next_Vl_row
    
    # Step 3: Combined V 계산
    # V = max(Vh, Vl - z)
    masked_z = (mask * z)[:, None]
    V_row = jnp.maximum(jnp.max(Vhs_row, axis=-1), Vl_row - masked_z)
    
    # Step 4: GAE 계산
    # Vhs_row, Vl_row, V_row를 concatenate하여 GAE 계수와 곱함
    cat_V_row = jnp.concatenate([Vhs_row, Vl_row[:, :, None], V_row[:, :, None]], axis=-1)
    Qs_GAE = ei.einsum(cat_V_row, gae_coeffs, "Tp1 na nhp2, Tp1 -> na nhp2")
    
    # GAE 계수 업데이트: [1] -> [λ 1-λ] -> [λ² λ(1-λ) 1-λ] -> ...
    gae_coeffs = jnp.roll(gae_coeffs, 1)
    gae_coeffs = gae_coeffs.at[0].set(gae_lambda ** (ii + 1))
    gae_coeffs = gae_coeffs.at[1].set((gae_lambda ** ii) * (1 - gae_lambda))
    
    return (Vhs_row, Vl_row, gae_coeffs), Qs_GAE
```

**핵심 포인트**:
1. **Backward DP**: 마지막 타임스텝(`T`)부터 역순으로 계산 (`reverse=True`)
2. **Vh 업데이트**: `Vh = max(cost, γ·Vh_next)` - Constraint value는 non-decreasing
3. **Vl 업데이트**: `Vl = -reward + γ·Vl_next` - Standard Bellman equation
4. **Combined V**: `V = max(max(Vh), Vl - z)` - Epigraph form의 핵심
5. **GAE**: 모든 미래 타임스텝의 value를 가중합하여 Q 함수 근사

**반환값**:
- `Qhs_GAEs`: `(T, n_agents, n_cost)` - Constraint Q 함수
- `Ql_GAEs`: `(T,)` - Reward Q 함수
- `Q_GAEs`: `(T, n_agents)` - Combined Q 함수

---

### 4. `compute_dec_efocp_V()` - Value 함수 계산

**위치**: `utils.py:118-121`

**역할**: 주어진 z 값에 대해 현재 상태의 value 계산

**입력**:
- `z: FloatScalar`: z 값 (scalar 또는 `(n_agents,)`)
- `Vhs: Array`: `(n_agents, n_cost)` - 각 에이전트, 각 constraint의 Vh 값
- `Vl: FloatScalar`: Vl 값 (scalar)

**수식**:
```python
V = jnp.maximum(Vhs.max(-1), (Vl - z))
```

**의미**:
- `Vhs.max(-1)`: 각 에이전트의 최대 constraint value
- `Vl - z`: Reward value에서 z를 뺀 값
- `max`: 두 값 중 큰 값 선택 (Epigraph form의 핵심)

**반환값**: `(n_agents,)` - 각 에이전트의 value

---

### 5. `update_policy()` - PPO Policy 업데이트

**위치**: `defmarl.py:500-543`

**역할**: PPO 알고리즘으로 Policy 네트워크 업데이트

**입력**:
- `policy_train_state: TrainState`: Policy 학습 상태
- `rollout: Rollout`: Rollout 데이터
- `bTa_A: Array`: `(batch, T, n_agents)` - Advantage 값
- `rnn_chunk_ids: Array`: RNN chunk 인덱스

**처리 단계**:

#### Step 1: Rollout을 RNN Chunk로 분할
```python
bcT_rollout = jax.tree.map(lambda x: x[:, rnn_chunk_ids], rollout)
rnn_state_inits = jnp.zeros_like(rollout.rnn_states[:, rnn_chunk_ids[:, 0]])
bcTa_A = bTa_A[:, rnn_chunk_ids]
```
- 긴 시퀀스를 `rnn_step` 크기의 chunk로 분할
- 각 chunk는 독립적으로 처리 (RNN 상태 초기화)

#### Step 2: 현재 정책의 Log Probability 계산
```python
bcTa_log_pis, bcTa_entropy, _, _ = jax.vmap(jax.vmap(
    ft.partial(self.scan_eval_action, actor_params=params)
))(bcT_rollout, rnn_state_inits, action_keys)
```
- `scan_eval_action`: 주어진 액션에 대한 log probability와 entropy 계산
- Rollout에서 사용된 액션을 현재 정책으로 재평가

#### Step 3: PPO Loss 계산
```python
bcTa_ratio = jnp.exp(bcTa_log_pis - bcT_rollout.log_pis)  # Importance ratio
loss_policy1 = bcTa_ratio * bcTa_A
loss_policy2 = jnp.clip(bcTa_ratio, 1 - self.clip_eps, 1 + self.clip_eps) * bcTa_A
loss_policy = jnp.maximum(loss_policy1, loss_policy2).mean()
```
- **Importance ratio**: `exp(log_π_new - log_π_old)`
- **Clipped surrogate loss**: `min(ratio·A, clip(ratio, 1-ε, 1+ε)·A)`
- Clipping으로 정책 업데이트가 너무 크지 않도록 제한

#### Step 4: Entropy 보너스 추가
```python
policy_loss = loss_policy - self.coef_ent * mean_entropy
```
- Entropy를 빼는 이유: Loss를 최소화하므로, entropy를 빼면 exploration을 장려

#### Step 5: Gradient 계산 및 업데이트
```python
grad, policy_info = jax.grad(get_loss, has_aux=True)(policy_train_state.params)
grad, grad_norm = compute_norm_and_clip(grad, self.max_grad_norm)
policy_train_state = policy_train_state.apply_gradients(grads=grad)
```

**반환값**: 업데이트된 학습 상태와 학습 정보

---

### 6. `update_value()` - Critic & Vh 네트워크 업데이트

**위치**: `defmarl.py:545-588`

**역할**: Critic (Vl)과 Vh 네트워크를 동시에 업데이트

**입력**:
- `critic_train_state, Vh_train_state`: 학습 상태
- `rollout: Rollout`: Rollout 데이터
- `bT_Ql: Array`: `(batch, T)` - Target Ql 값
- `bTah_Qh: Array`: `(batch, T, n_agents, n_cost)` - Target Qh 값
- `rnn_states_Vl, rnn_states_Vh`: RNN 상태
- `rnn_chunk_ids: Array`: RNN chunk 인덱스

**처리 단계**:

#### Step 1: Rollout을 RNN Chunk로 분할
```python
bcT_rollout = jax.tree.map(lambda x: x[:, rnn_chunk_ids], rollout)
Vl_rnn_state_inits = jnp.zeros_like(rnn_states_Vl[:, rnn_chunk_ids[:, 0]])
Vh_rnn_state_inits = jnp.zeros_like(rnn_states_Vh[:, rnn_chunk_ids[:, 0]])
bcT_Ql = bT_Ql[:, rnn_chunk_ids]
bcTah_Qh = bTah_Qh[:, rnn_chunk_ids]
```

#### Step 2: 현재 Value 함수 값 계산
```python
(bcT_Vl, bcTah_Vh), _, _ = jax.vmap(jax.vmap(
    ft.partial(self.scan_value, critic_params=critic_params, Vh_params=Vh_params)))(
    bcT_rollout, Vl_rnn_state_inits, Vh_rnn_state_inits
)
```

#### Step 3: L2 Loss 계산
```python
loss_Vl = optax.l2_loss(bcT_Vl, bcT_Ql).mean()
loss_Vh = optax.l2_loss(bcTah_Vh, bcTah_Qh).mean()
total_loss = loss_Vl + loss_Vh
```
- **Critic Loss**: 예측 Vl과 target Ql의 차이
- **Vh Loss**: 예측 Vh와 target Qh의 차이
- 두 loss를 합하여 동시에 최적화

#### Step 4: Gradient 계산 및 업데이트
```python
(grad_Vl, grad_Vh), value_info = jax.grad(get_loss, argnums=(0, 1), has_aux=True)(
    critic_train_state.params, Vh_train_state.params)
grad_Vl, grad_Vl_norm = compute_norm_and_clip(grad_Vl, self.max_grad_norm)
grad_Vh, grad_Vh_norm = compute_norm_and_clip(grad_Vh, self.max_grad_norm)
critic_train_state = critic_train_state.apply_gradients(grads=grad_Vl)
Vh_train_state = Vh_train_state.apply_gradients(grads=grad_Vh)
```

**반환값**: 업데이트된 학습 상태와 학습 정보

---

### 7. `RootFinder` - z 최적화

**위치**: `module/root_finder.py:16-74`

**역할**: 각 에이전트의 최적 z 값을 찾기 위한 root finding

**초기화 파라미터**:
- `z_min: float`: z의 최소값 (보통 `-env.reward_max`)
- `z_max: float`: z의 최대값 (보통 `-env.reward_min`)
- `n_agent: int`: 에이전트 수
- `h_tgt: float = -0.2`: 목표 constraint 값 (안전 임계값)
- `h_eps: float = 1e-2`: 허용 오차
- `n_iters: int = 20`: Root finding 최대 반복 횟수
- `z_comm: bool = False`: 에이전트 간 z 통신 여부

#### `get_dec_opt_z(Vh_fn, graph)`
- **역할**: 모든 에이전트의 최적 z 값을 계산
- **입력**:
  - `Vh_fn: Callable`: Vh 함수 (z를 입력받아 Vh 값 반환)
  - `graph: GraphsTuple`: 현재 환경 그래프
- **동작**:
  1. 각 에이전트에 대해 `get_opt_z` 호출
  2. `z_comm=True`이면 그래프 구조를 이용해 에이전트 간 z 값 통신
     - 각 에이전트는 이웃 에이전트의 z 값 중 최대값을 자신의 z와 비교하여 업데이트
  3. 최종 z 값과 RNN 상태 반환

#### `get_opt_z(Vh_fn, i_agent)`
- **역할**: 특정 에이전트의 최적 z 값 계산
- **입력**:
  - `Vh_fn: Callable`: Vh 함수
  - `i_agent: IntScalar`: 에이전트 인덱스
- **Root Finding 함수**:
  ```python
  def z_root_fn(z):
      h_Vh, _ = Vh_fn(z[None, None].repeat(self.n_agent, axis=0))
      h_Vh = h_Vh[i_agent]
      Vh = h_Vh.max()  # 최대 constraint value
      root = -(Vh - self.h_tgt)  # Vh = h_tgt를 만족하는 z 찾기
      return root
  ```
- **의미**: `Vh(z) = h_tgt`를 만족하는 z를 찾음
  - `Vh < h_tgt`: 안전 (constraint 만족)
  - `Vh > h_tgt`: 불안전 (constraint 위반)
- **알고리즘**: Chandrupatla root finding 알고리즘 사용
- **Edge Cases 처리**:
  ```python
  both_safe = (init_state.y1 > 0) & (init_state.y2 > 0)
  both_unsafe = (init_state.y1 < 0) & (init_state.y2 < 0)
  opt_z = jnp.where(both_safe, self.z_min, 
                   jnp.where(both_unsafe, self.z_max, opt_z))
  ```
  - `z_min`과 `z_max` 모두 안전하면 → `z_min` 선택 (최소 z)
  - 모두 불안전하면 → `z_max` 선택 (최대 z)
  - 그 외 → Root finding 결과 사용

**반환값**: 최적 z 값 (scalar)

---

### 8. `scan_eval_action()` - 액션 평가

**위치**: `defmarl.py:483-498`

**역할**: 주어진 액션에 대한 현재 정책의 log probability와 entropy 계산

**입력**:
- `rollout: Rollout`: Rollout 데이터
- `init_rnn_state: Array`: RNN 초기 상태
- `action_keys: PRNGKey`: 랜덤 키
- `actor_params: Params`: Actor 파라미터

**동작**:
```python
def body_(rnn_state, inp):
    graph, z, key, action = inp
    log_pi, entropy, new_rnn_state = self.policy.eval_action(
        actor_params, graph, action, rnn_state, key, z)
    return new_rnn_state, (log_pi, entropy, rnn_state)
```

**용도**: PPO 업데이트 시 importance ratio 계산
- Rollout에서 사용된 액션을 현재 정책으로 재평가
- `ratio = exp(log_π_new - log_π_old)`

**반환값**:
- `Ta_log_pis`: `(T, n_agents)` - Log probability
- `Ta_entropies`: `(T, n_agents)` - Entropy
- `rnn_states`: RNN 상태 시퀀스
- `final_rnn_state`: 최종 RNN 상태

---

## 학습 데이터 흐름

### 전체 학습 루프

```
1. collect(params, key)
   └─> rollout_fn (환경과 상호작용)
       └─> step() (액션 샘플링)
           └─> policy.sample_action()
       └─> env.step() (환경 업데이트)
       └─> get_opt_z() (z 최적화, RootFinder 사용)
   └─> Rollout 반환

2. update(rollout, step)
   └─> update_inner() [핵심 학습 함수]
       ├─> scan_value() 
       │   └─> critic.get_value() (Vl 계산)
       │   └─> Vh.get_value() (Vh 계산)
       ├─> compute_dec_efocp_gae() [핵심 GAE 계산]
       │   └─> Backward DP로 Q 함수 계산
       ├─> compute_dec_efocp_V() (현재 V 계산)
       ├─> Advantage 계산 및 정규화
       ├─> update_value() (Critic, Vh 업데이트)
       │   └─> L2 loss로 value 함수 학습
       └─> update_policy() (Policy 업데이트)
           └─> PPO clipped surrogate loss
   └─> 학습 정보 반환
```

### Rollout 데이터 구조

`Rollout`은 다음을 포함합니다:
- `graph: GraphsTuple`: 환경 그래프 시퀀스 `(T,)`
- `actions: Array`: 액션 시퀀스 `(T, n_agents, action_dim)`
- `rewards: Array`: 보상 시퀀스 `(T,)` (scalar, 모든 에이전트 공유)
- `costs: Array`: 비용 시퀀스 `(T, n_agents, n_cost)` (에이전트별, constraint별)
- `dones: Array`: 종료 플래그 `(T,)`
- `log_pis: Array`: Log probability `(T, n_agents)`
- `rnn_states: Array`: RNN 상태 `(T, ...)`
- `zs: Array`: z 값 `(T, n_agents, 1)` (에이전트별 z)
- `next_graph: GraphsTuple`: 마지막 다음 상태

---


---

## 참고사항

### JIT 컴파일
- `update_inner`: JIT 컴파일됨
- `rollout_fn`: JIT 컴파일됨
- `get_init_graph`: JIT 컴파일됨

### 메모리 관리
- `use_prev_init=True`일 때 이전 rollout의 마지막 상태를 재사용
- RNN chunking으로 긴 시퀀스 처리

### Gradient Clipping
- 모든 네트워크에 `max_grad_norm` 적용
- NaN/Inf 체크 및 처리

---

## 사용 예시

```python
# 초기화
algo = DefMARL(
    env=env,
    node_dim=7,
    edge_dim=4,
    state_dim=4,
    action_dim=2,
    n_agents=5,
    # ... 하이퍼파라미터
)

# 학습 루프
for step in range(max_steps):
    # 데이터 수집
    rollout = algo.collect(algo.params, key)
    
    # 업데이트
    info = algo.update(rollout, step)
    
    # 로깅
    print(f"Step {step}: {info}")
```

---

## Pseudocode와 코드 매핑

이 섹션은 Centralized Training process와 Distributed execution process의 pseudocode가 DefMARL 코드의 어느 메서드/함수에 매핑되는지 설명합니다.

### Centralized Training Process

#### 1. Initialize: Policy NN π_θ, cost value function NN V_φ^l, constraint value function NN V_ψ^h

**코드 위치**: `defmarl.py:__init__()` (line 29-257)

**매핑**:
- **Policy NN π_θ**: `self.policy = PPOPolicy(...)` (line 109-120)
  - 초기화: `policy_params = self.policy.dist.init(...)` (line 135-137)
  - TrainState: `self.policy_train_state` (line 140-144)

- **Cost value function NN V_φ^l**: `self.critic = ValueNet(...)` (line 147-159)
  - 초기화: `critic_params = self.critic.net.init(...)` (line 173-174)
  - TrainState: `self.critic_train_state` (line 177-181)

- **Constraint value function NN V_ψ^h**: `self.Vh = ValueNet(...)` (line 184-197)
  - 초기화: `Vh_params = self.Vh.net.init(...)` (line 210)
  - TrainState: `self.Vh_train_state` (line 213-217)

---

#### 2. Randomly sampling initial conditions x^0, and the initial z^0 ∈ [z_min, z_max]

**코드 위치**: `trainer/utils.py:rollout()` (line 43-53)

**매핑**:
```python
# Initial state sampling
key_x0, key_z0, key = jax.random.split(key, 3)
if init_graph is None:
    init_graph = env.reset(key_x0)  # x^0 sampling

# Initial z sampling
z0 = jax.random.uniform(key_z0, (1, 1), 
                       minval=-env.reward_max,  # z_min
                       maxval=-env.reward_min)  # z_max

# Additional z sampling strategies
z_key, key = jax.random.split(key, 2)
rng = jax.random.uniform(z_key, (1, 1))
z0 = jnp.where(rng > 0.7, -env.reward_max, z0)  # use z min
z0 = jnp.where(rng < 0.2, -env.reward_min, z0)  # use z max

z0 = jnp.repeat(z0, env.num_agents, axis=0)  # Broadcast to all agents
```

**호출 경로**: `defmarl.py:collect()` → `self.rollout_fn()` → `trainer/utils.py:rollout()`

---

#### 3. Use π_θ to sample trajectories {x^0, …, x^T}, with z dynamics

**코드 위치**: `trainer/utils.py:rollout()` (line 55-69)

**매핑**:
```python
def body(data, key_):
    graph, rnn_state, z = data
    
    # Sample action using policy π_θ
    action, log_pi, new_rnn_state = actor(graph, z, rnn_state, key_)
    # actor = defmarl.step() = policy.sample_action()
    
    # Environment step
    next_graph, reward, cost, done, info = env.step(graph, action)
    
    # z dynamics: z^{k+1} = z^k - l(x^k, π(x^k))
    # In code: z_next = (z + reward) / gamma
    # Note: reward is negative of l, so z + reward = z - l
    z_next = (z + reward) / gamma
    z_next = jnp.clip(z_next, -env.reward_max, -env.reward_min)
    
    return ((next_graph, new_rnn_state, z_next),
            (graph, action, rnn_state, reward, cost, done, log_pi, next_graph, z))

# Scan over trajectory
keys = jax.random.split(key, env.max_episode_steps)
_, (graphs, actions, rnn_states, rewards, costs, dones, log_pis, next_graphs, zs) = (
    jax.lax.scan(body, (init_graph, init_rnn_state, z0), keys, 
                 length=env.max_episode_steps))
```

**핵심 포인트**:
- **Policy sampling**: `actor(graph, z, rnn_state, key_)` → `defmarl.step()` → `policy.sample_action()`
- **z dynamics**: `z_next = (z + reward) / gamma` 
  - Pseudocode의 `z^{k+1} = z^k - l(x^k, π(x^k))`와 동일 (reward = -l)
  - `V(x^k, z^k; π) = max{h(x^k), V(x^{k+1}, z^{k+1}; π)}`는 GAE 계산에서 처리

**호출 경로**: `defmarl.py:collect()` → `self.rollout_fn()` → `trainer/utils.py:rollout()`

---

#### 4. Calculate the cost value function V_φ^l(x,z) and the constraint value function V_ψ^h(o_i, z)

**코드 위치**: `defmarl.py:scan_value()` (line 381-411)

**매핑**:
```python
def body_(rnn_state, inp):
    graph, z = inp
    rnn_state_Vl, rnn_state_Vh = rnn_state
    
    # Calculate V_φ^l(x, z)
    # Note: Critic uses z[0] (same z for all agents)
    value, new_rnn_state_V = self.critic.get_value(
        critic_params, graph, rnn_state_Vl, z[0][None, :])
    
    # Calculate V_ψ^h(o_i, z)
    # Note: Vh uses per-agent z
    value_h, new_rnn_state_Vh = self.Vh.get_value(
        Vh_params, graph, rnn_state_Vh, z)
    
    return (new_rnn_state_V, new_rnn_state_Vh), (value, value_h, ...)

# Scan over all timesteps
(final_rnn_state_Vl, final_rnn_state_Vh), (T_Vl, Tah_Vh, ...) = (
    jax.lax.scan(body_, (init_rnn_state_Vl, init_rnn_state_Vh), (graphs, zs)))
```

**핵심 포인트**:
- **V_φ^l(x, z)**: `self.critic.get_value()` → `(T,)` shape
- **V_ψ^h(o_i, z)**: `self.Vh.get_value()` → `(T, n_agents, n_cost)` shape
- 각 타임스텝, 각 에이전트, 각 constraint에 대한 value 계산

**호출 경로**: `defmarl.py:update_inner()` → `scan_value()`

---

#### 5. Calculate GAE with the total value function

**코드 위치**: 
- `defmarl.py:update_inner()` (line 447-454)
- `algo/utils.py:compute_dec_efocp_gae()` (line 40-115)
- `algo/utils.py:compute_dec_efocp_V()` (line 118-121)

**매핑**:

**Step 5.1: Total value function 계산**
```python
# In update_inner()
bTa_V = jax_vmap(jax_vmap(compute_dec_efocp_V))(
    rollout.zs.squeeze(-1)[:, :, 0],  # z values
    bTah_Vh,  # V_ψ^h(o_i, z)
    bT_Vl     # V_φ^l(x, z)
)
```

**`compute_dec_efocp_V()` 함수** (`utils.py:118-121`):
```python
def compute_dec_efocp_V(z: FloatScalar, Vhs: Float[Array, "a nh"], Vl: FloatScalar) -> FloatScalar:
    # V_i(x^τ, z; π) = max{V_i^h(o_i^τ; π), V^l(x^τ; π) - z}
    return jnp.maximum(Vhs.max(-1), (Vl - z))
    # Vhs.max(-1): max over constraints for each agent
    # Vl - z: reward value minus z
    # max: take maximum (Epigraph form)
```

**Step 5.2: GAE 계산**
```python
# In update_inner()
bTah_Qh, bT_Ql, bTa_Q = jax.vmap(
    ft.partial(compute_dec_efocp_gae, disc_gamma=self.gamma, gae_lambda=self.gae_lambda)
)(Tah_hs=rollout.costs,      # h(x^k)
  T_l=-rollout.rewards,      # l(x^k) = -reward
  T_z=rollout.zs.squeeze(-1)[:, :, 0],  # z^k
  Tp1ah_Vh=bTp1ah_Vh,        # V_ψ^h(o_i^{k+1}, z^{k+1})
  Tp1_Vl=bTp1_Vl)            # V_φ^l(x^{k+1}, z^{k+1})
```

**`compute_dec_efocp_gae()` 함수** (`utils.py:40-115`):
- **Backward DP**: 마지막 타임스텝부터 역순으로 계산
- **Vh 업데이트**: `Vh = max(cost, γ·Vh_next)` (line 76)
- **Vl 업데이트**: `Vl = -reward + γ·Vl_next` (line 78)
- **Combined V**: `V = max(max(Vh), Vl - z)` (line 82)
- **GAE 적용**: 모든 미래 타임스텝의 value를 가중합하여 Q 함수 계산 (line 86)

**핵심 포인트**:
- Total value function: `V(x^τ, z; π) = max_i V_i(x^τ, z; π)`
  - `V_i(x^τ, z; π) = max{V_i^h(o_i^τ; π), V^l(x^τ; π) - z}`
- GAE는 Backward DP를 사용하여 계산

**호출 경로**: `defmarl.py:update_inner()` → `compute_dec_efocp_gae()`

---

#### 6. Update the value functions V_φ^l and V_ψ^h using TD error

**코드 위치**: `defmarl.py:update_value()` (line 545-588)

**매핑**:
```python
def get_loss(critic_params, Vh_params):
    # Calculate current value predictions
    (bcT_Vl, bcTah_Vh), _, _ = jax.vmap(jax.vmap(
        ft.partial(self.scan_value, critic_params=critic_params, Vh_params=Vh_params)))(
        bcT_rollout, Vl_rnn_state_inits, Vh_rnn_state_inits
    )
    
    # TD error: target Q - predicted V
    loss_Vl = optax.l2_loss(bcT_Vl, bcT_Ql).mean()      # V_φ^l update
    loss_Vh = optax.l2_loss(bcTah_Vh, bcTah_Qh).mean()  # V_ψ^h update
    
    return loss_Vl + loss_Vh, info

# Gradient descent
(grad_Vl, grad_Vh), value_info = jax.grad(get_loss, argnums=(0, 1), has_aux=True)(
    critic_train_state.params, Vh_train_state.params)
critic_train_state = critic_train_state.apply_gradients(grads=grad_Vl)
Vh_train_state = Vh_train_state.apply_gradients(grads=grad_Vh)
```

**핵심 포인트**:
- **Target**: GAE로 계산된 Q 값 (`bT_Ql`, `bTah_Qh`)
- **Prediction**: 현재 네트워크의 V 값 (`bcT_Vl`, `bcTah_Vh`)
- **Loss**: L2 loss (TD error의 제곱)
- **Update**: Gradient descent로 네트워크 파라미터 업데이트

**호출 경로**: `defmarl.py:update_inner()` → `update_value()`

---

#### 7. Update the z-conditioned policy π_θ(·, z) using PPO loss

**코드 위치**: `defmarl.py:update_policy()` (line 500-543)

**매핑**:
```python
def get_loss(params):
    # Evaluate current policy on rollout actions
    bcTa_log_pis, bcTa_entropy, _, _ = jax.vmap(jax.vmap(
        ft.partial(self.scan_eval_action, actor_params=params)
    ))(bcT_rollout, rnn_state_inits, action_keys)
    
    # Importance ratio
    bcTa_ratio = jnp.exp(bcTa_log_pis - bcT_rollout.log_pis)
    
    # PPO clipped surrogate loss
    loss_policy1 = bcTa_ratio * bcTa_A
    loss_policy2 = jnp.clip(bcTa_ratio, 1 - self.clip_eps, 1 + self.clip_eps) * bcTa_A
    loss_policy = jnp.maximum(loss_policy1, loss_policy2).mean()
    
    # Entropy bonus for exploration
    mean_entropy = bcTa_entropy.mean()
    policy_loss = loss_policy - self.coef_ent * mean_entropy
    
    return policy_loss, info

# Gradient descent
grad, policy_info = jax.grad(get_loss, has_aux=True)(policy_train_state.params)
policy_train_state = policy_train_state.apply_gradients(grads=grad)
```

**핵심 포인트**:
- **Importance ratio**: `exp(log_π_new - log_π_old)`
- **PPO loss**: `min(ratio·A, clip(ratio, 1-ε, 1+ε)·A)`
- **Entropy bonus**: Exploration을 위한 entropy 추가
- **z-conditioned**: Policy는 z를 입력으로 받음 (`policy.sample_action(graph, rnn_state, key, z)`)

**호출 경로**: `defmarl.py:update_inner()` → `update_policy()`

---

### Distributed Execution Process

#### 1. Get z_i for each agent by solving the distributed EF-MASOCP outer problem

**코드 위치**: 
- `defmarl.py:get_opt_z()` (line 295-305)
- `module/root_finder.py:get_dec_opt_z()` (line 34-51)
- `module/root_finder.py:get_opt_z()` (line 53-73)

**매핑**:

**`get_opt_z()` 메서드** (`defmarl.py:295-305`):
```python
def get_opt_z(self, graph: GraphsTuple, Vh_rnn_state: Array, params: Optional[Params] = None):
    def fn_(Vh_params, obs, rnn_state):
        Vh_fn = ft.partial(self.Vh_train_state.apply_fn, Vh_params, obs, rnn_state)
        return self.root_finder.get_dec_opt_z(Vh_fn, obs)
    
    return jax.jit(fn_)(params["Vh"], graph, Vh_rnn_state)
```

**`get_dec_opt_z()` 메서드** (`root_finder.py:34-51`):
```python
def get_dec_opt_z(self, Vh_fn: Callable, graph: GraphsTuple):
    agent_idx = jnp.arange(self.n_agent)
    solve_fn = ft.partial(self.get_opt_z, Vh_fn)
    
    # Solve for each agent independently
    opt_z = jax_vmap(solve_fn)(agent_idx)  # (n_agent,)
    
    # z communication (if enabled)
    if self.z_comm:
        opt_z = jax.lax.fori_loop(0, self.n_agent, z_comm_, opt_z)
    
    opt_z = opt_z[:, None]  # (n_agent, 1)
    _, rnn_states = Vh_fn(opt_z)
    return opt_z, rnn_states
```

**`get_opt_z()` 메서드** (`root_finder.py:53-73`):
```python
def get_opt_z(self, Vh_fn: Callable, i_agent: IntScalar) -> FloatScalar:
    def z_root_fn(z):
        h_Vh, _ = Vh_fn(z[None, None].repeat(self.n_agent, axis=0))
        h_Vh = h_Vh[i_agent]
        Vh = h_Vh.max()  # max over constraints
        # Root finding: V_i^h(o_i; π(·, z')) ≤ 0 (h_tgt = -0.2)
        root = -(Vh - self.h_tgt)
        return root
    
    # Solve: z_i = min_{z'} z', s.t. V_i^h(o_i; π(·, z')) ≤ 0
    solver = Chandrupatla(z_root_fn, n_iters=self.n_iters, init_t=0.5)
    opt_z, _, init_state = solver.run(self.z_min, self.z_max)
    
    # Edge cases
    both_safe = (init_state.y1 > 0) & (init_state.y2 > 0)
    both_unsafe = (init_state.y1 < 0) & (init_state.y2 < 0)
    opt_z = jnp.where(both_safe, self.z_min, 
                     jnp.where(both_unsafe, self.z_max, opt_z))
    
    return opt_z
```

**핵심 포인트**:
- **문제**: `z_i = min_{z'} z', s.t. V_i^h(o_i; π(·, z')) ≤ 0`
- **Root finding**: `V_i^h(o_i; π(·, z')) = h_tgt`를 만족하는 z 찾기
- **알고리즘**: Chandrupatla root finding method
- **각 에이전트 독립적으로 계산**: `jax_vmap(solve_fn)(agent_idx)`

**호출 경로**: 평가/실행 시 → `defmarl.get_opt_z()` → `root_finder.get_dec_opt_z()` → `root_finder.get_opt_z()`

---

#### 2. If z communication enabled, communicate z_j and reach consensus z = max_j z_j

**코드 위치**: `module/root_finder.py:get_dec_opt_z()` (line 39-47)

**매핑**:
```python
def z_comm_(i, z):
    # Get z values from connected agents (via graph structure)
    z_map = jtu.tree_map(lambda n: safe_get(n, graph.senders, fill_value=-jnp.inf), z)
    # Segment max: find max z in each agent's neighborhood
    max_z = jraph.segment_max(z_map, segment_ids=graph.receivers, 
                              num_segments=graph.nodes.shape[0])
    max_z = max_z[:self.n_agent]
    # Consensus: z = max_j z_j
    z = jnp.maximum(max_z, z)
    return z

if self.z_comm:
    opt_z = jax.lax.fori_loop(0, self.n_agent, z_comm_, opt_z)
```

**핵심 포인트**:
- **Graph-based communication**: 그래프 구조를 이용해 이웃 에이전트의 z 값 수신
- **Consensus**: 각 에이전트는 자신과 이웃의 z 값 중 최대값 선택
- **Iterative**: `fori_loop`로 여러 번 반복하여 consensus 도달

**조건**: `self.z_comm = True`일 때만 활성화

---

#### 3. Get decentralized policy π_i(·) = π_θ(·, z_i)

**코드 위치**: `defmarl.py:act()` (line 307-317)

**매핑**:
```python
def act(self, graph: GraphsTuple, z: Array, rnn_state: Array, params: Optional[Params] = None):
    if params is None:
        params = self.params
    # Get action from policy conditioned on z_i
    action, rnn_state = self.policy.get_action(params["policy"], graph, rnn_state, z)
    return action, rnn_state
```

**핵심 포인트**:
- **z-conditioned policy**: Policy는 z 값을 입력으로 받음
- **Decentralized**: 각 에이전트는 자신의 z_i를 사용
- **Observation-based**: Policy는 각 에이전트의 observation (graph)만 사용

**호출 경로**: 평가/실행 시 → `defmarl.act()`

---

#### 4. Execute control u_i^k = π_i(o_i^k)

**코드 위치**: `trainer/utils.py:test_rollout()` (line 99-113)

**매핑**:
```python
def body(data, key_):
    graph, actor_rnn_state, Vh_rnn_state = data
    
    # Step 1: Get z_i for each agent
    if z_fn is not None:
        z, Vh_rnn_state = z_fn(graph, Vh_rnn_state)  # get_opt_z()
        # Optional: use max z for all agents (if communication)
        z_max = np.max(z, axis=0)
        z = jnp.repeat(z_max[None], env.num_agents, axis=0)
    else:
        z = z0
    
    # Step 2: Execute control u_i^k = π_i(o_i^k)
    # This is the actual policy execution: sample action from π_i(·) = π_θ(·, z_i)
    if not stochastic:
        action, actor_rnn_state = actor(graph, z, actor_rnn_state)  # act() → policy.get_action()
    else:
        action, actor_rnn_state = actor(graph, z, actor_rnn_state, key_)  # act() → policy.sample_action()
    
    # Step 3: Apply action to environment
    next_graph, reward, cost, done, info = env.step(graph, action)
    
    return ((next_graph, actor_rnn_state, Vh_rnn_state), ...)
```

**핵심 포인트**:
- **z 계산**: `z_fn = get_opt_z()`로 각 에이전트의 z_i 계산
- **Execute control u_i^k = π_i(o_i^k)**: `actor(graph, z, actor_rnn_state)` 
  - 이것이 pseudocode의 "Execute control u_i^k = π_i(o_i^k)"에 해당
  - `actor = defmarl.act()` → `policy.get_action(graph, rnn_state, z)`
  - z-conditioned policy에서 액션을 샘플링하는 단계
- **환경 적용**: `env.step(graph, action)` - 샘플링된 액션을 환경에 적용

**호출 경로**: 평가 시 → `trainer/utils.py:test_rollout()` → `actor()` (defmarl.act) → `policy.get_action()`

---

## Pseudocode 요약 매핑

| Pseudocode 단계 | 코드 위치 | 주요 메서드/함수 |
|----------------|----------|----------------|
| **Training: Initialize** | `defmarl.py:__init__()` | `PPOPolicy`, `ValueNet` (Critic, Vh) 초기화 |
| **Training: Sample x^0, z^0** | `trainer/utils.py:rollout()` | `env.reset()`, `jax.random.uniform()` |
| **Training: Sample trajectories** | `trainer/utils.py:rollout()` | `actor()` → `defmarl.step()` → `policy.sample_action()` |
| **Training: z dynamics** | `trainer/utils.py:rollout()` | `z_next = (z + reward) / gamma` |
| **Training: Calculate V^l, V^h** | `defmarl.py:scan_value()` | `critic.get_value()`, `Vh.get_value()` |
| **Training: Calculate GAE** | `defmarl.py:update_inner()` | `compute_dec_efocp_gae()`, `compute_dec_efocp_V()` |
| **Training: Update V^l, V^h** | `defmarl.py:update_value()` | L2 loss, gradient descent |
| **Training: Update π_θ** | `defmarl.py:update_policy()` | PPO clipped surrogate loss |
| **Execution: Solve for z_i** | `defmarl.py:get_opt_z()` | `root_finder.get_dec_opt_z()` → `root_finder.get_opt_z()` |
| **Execution: z communication** | `root_finder.py:get_dec_opt_z()` | `z_comm_()` (if `z_comm=True`) |
| **Execution: Get policy** | `defmarl.py:act()` | `policy.get_action(graph, rnn_state, z)` |
| **Execution: Execute control** | `trainer/utils.py:test_rollout()` | `actor(graph, z, rnn_state)` → `policy.get_action()` (u_i^k = π_i(o_i^k)) |

---

## 파일 위치

- **메인 파일**: `def-marl/defmarl/algo/defmarl.py`
- **Base 클래스**: `def-marl/defmarl/algo/base.py`
- **유틸리티**: `def-marl/defmarl/algo/utils.py`
- **모듈**: `def-marl/defmarl/algo/module/`
- **Rollout 함수**: `def-marl/defmarl/trainer/utils.py`
- **Root Finder**: `def-marl/defmarl/algo/module/root_finder.py`

