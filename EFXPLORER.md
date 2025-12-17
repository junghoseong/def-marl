# EFXplorer: Conservative Exploration Framework 구현 가이드

## 개요

EFXplorer는 entropy 기반 intrinsic reward를 사용한 Conservative Exploration Framework를 구현합니다. 기존 Safe MARL (DefMARL)과 대치되는 방법론으로, exploration을 constraint로 다루고 extrinsic performance를 보장합니다.

**핵심 원칙**: Vi/Intrinsic value는 사용하지 않습니다. Ve (extrinsic value), Ae (extrinsic advantage), Δ_team_int (team-level intrinsic reward), z (budget)만 사용합니다.

---

## 1. 수학적 프레임워크

### 1.1 최적화 문제

**Inner Loop (Policy Update)**:
$$
\max_{\pi\in\Pi} J_\text{int}(\pi) \quad \text{s.t.} \quad J_\text{ext}(\pi) \geq J_\text{ext}(\pi_{\text{old}})
$$

**Outer Loop (Budget Update)**:
$$
\max_z z \quad \text{s.t.}\quad \max_\pi \min \{J_\text{ext}(\pi)-J_\text{ext}(\pi^\text{old}), J_\text{int}(\pi)-z\}\geq0
$$

### 1.2 Extrinsic Advantage

$$
A^\text{ext}(s_t, a_t) = Q^{\text{ext}}(s_t, a_t) - V^\text{ext}(s_t) = r_t^\text{ext} + \gamma \mathbb{E}_{s_{t+1}}[V^\text{ext}(s_{t+1})] - V^\text{ext}(s_t)
$$

### 1.3 S_t 정의 및 Recursive Form

$$
S_t(s_t, a_t) := R^\text{int}_t - z_t = \sum_{k=t}^{T} \Delta_k^\text{team-int} - z_t
$$

여기서 $\Delta_t^\text{team-int} = \sum_i r_{i,t}^\text{int}$ (team-level intrinsic reward)

**Recursive Form** (discount 없음):
$$
S_t = R_t^\text{int} - z_t = S_{t+1}
$$

**z-dynamics**:
$$
z_{t+1} = z_t - \Delta_t^\text{team-int}
$$
(단, clip: $z \in [-R_\text{max}, -R_\text{min}]$)

### 1.4 Final Advantage (DP Formulation)

**Truncated Version**:
$$
A_t^{(T)} = \min\{A_t^\text{ext}, A_{t+1}^\text{ext}, ..., A_T^\text{ext}, S_t\}
$$

**DP Equation**:
$$
A_t^{(T)} = \min\{A_t^\text{ext}, A_{t+1}^{(T)}\}
$$

**Terminal Condition**:
$$
A_T^{(T)} := A_T^\text{ext} \wedge S_T = \min\{A_T^\text{ext}, S_T\}
$$

---

## 2. 구현 구조

### 2.1 데이터 구조

**Rollout** (`trainer/data.py`):
- `intrinsic_rewards`: `(T, n_agents)` - Per-agent intrinsic rewards $r_{i,t}^\text{int} = -\log \pi_\theta(a_i|s_i) \cdot c$
- `rewards`: `(T, n_agents)` - Extrinsic rewards
- `zs`: `(T, n_agents, 1)` - Budget variables (shared z per agent slot)

**Team-level intrinsic reward**:
- $\Delta_t^\text{team-int} = \sum_i r_{i,t}^\text{int}$ (rollout에서 계산)

### 2.2 네트워크 구조

**Ve (Critic)**: Global extrinsic value `(T,)`
- `ValueNet(use_ef=False, decompose=False)` - 기존 DefMARL의 Vl과 동일
- Global value만 계산 (agent-wise 아님)

**Vi 네트워크**: ❌ **사용하지 않음**

---

## 3. 핵심 함수 구현

### 3.1 `compute_conservative_exploration_gae` 함수

**위치**: `defmarl/algo/utils.py:126-198`

**시그니처**:
```python
def compute_conservative_exploration_gae(
    Tah_Aext: Float[Array, "T a"],      # (T, n_agents) - Extrinsic advantages
    T_Delta_team_int: Float[Array, "T"], # (T,) - Team intrinsic rewards
    T_z: Float[Array, "T a"],           # (T, n_agents) - Budget variables
) -> Float[Array, "T a"]:
```

**구현 로직**:
1. **S_t 계산** (backward scan):
   - $R_t^\text{int} = \sum_{k=t}^T \Delta_k^\text{team-int}$ (cumulative sum backward)
   - $S_t = R_t^\text{int} - z_t$ (per agent)

2. **A_t^{(T)} 계산** (backward DP):
   - Terminal: $A_T^{(T)} = \min\{A_T^\text{ext}, S_T\}$
   - Recurrence: $A_t^{(T)} = \min\{A_t^\text{ext}, A_{t+1}^{(T)}\}$

**현재 상태**: ✅ 구현 완료

### 3.2 `compute_extrinsic_gae` 함수

**위치**: `defmarl/algo/utils.py:200-231`

**시그니처**:
```python
def compute_extrinsic_gae(
    values: Float[Array, "T"],          # (T,) - Global extrinsic values
    rewards: Float[Array, "T a"],       # (T, n_agents) - Extrinsic rewards
    next_values: Float[Array, "Tp1"],   # (T+1,) - Next values
    dones: Float[Array, "T"],           # (T,) - Done flags
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Float[Array, "T a"]:
```

**구현 로직**:
1. Rewards를 agent-wise로 sum: $r_t^\text{sum} = \sum_i r_{i,t}^\text{ext}$
2. Global GAE 계산: `compute_gae` 사용
3. Agent-wise로 broadcast: 모든 agent가 같은 advantage 사용

**현재 상태**: ✅ 구현 완료

---

## 4. Rollout 함수

### 4.1 `rollout_efxplorer` 함수

**위치**: `defmarl/trainer/utils.py:88-138`

**핵심 로직**:

```python
def body(data, key_):
    graph, rnn_state, z = data
    
    # Sample action
    action, log_pi, new_rnn_state = actor(graph, z, rnn_state, key_)
    
    # Environment step
    next_graph, reward, cost, done, info = env.step(graph, action)
    
    # Compute intrinsic reward (per-agent)
    intrinsic_reward = -log_pi * intrinsic_coef  # (n_agents,)
    
    # Compute team-level intrinsic reward
    Delta_team_int = intrinsic_reward.sum()  # Scalar
    
    # z-dynamics: z_{t+1} = z_t - Δ_t^team-int
    z_next = jnp.clip(z - Delta_team_int, -env.reward_max, -env.reward_min)
    
    return ((next_graph, new_rnn_state, z_next),
            (graph, action, rnn_state, reward, intrinsic_reward, cost, done, log_pi, next_graph, z))
```

**현재 상태**: ✅ 구현 완료 (line 123-125)

---

## 5. 알고리즘 클래스 (`efxplorer.py`)

### 5.1 `__init__` 메서드

**위치**: `efxplorer.py:30-216`

**하이퍼파라미터**:
- `intrinsic_coef`: Intrinsic reward coefficient (default: 1.0)
- `lr_z`: Budget z learning rate (default: 1e-7)

**네트워크 초기화**:
- ✅ Policy: `PPOPolicy` (z-conditioned)
- ✅ Critic (Ve): `ValueNet(use_ef=False, decompose=False)`
- ❌ Vi: **제거됨**

**현재 상태**: ✅ Vi 제거 완료

### 5.2 `scan_value` 메서드

**위치**: `efxplorer.py:321-361`

**시그니처**:
```python
def scan_value(
    self,
    rollout: Rollout,
    init_rnn_state_Ve: Array,
    critic_params: Params,
) -> Tuple[Tuple[Array], Tuple[Array], Tuple[Array]]:
    """
    Compute extrinsic (Ve) values only
    """
```

**반환값**:
- `T_Ve`: `(T,)` - Extrinsic values
- `rnn_states_Ve`: RNN states
- `final_rnn_state_Ve`: Final RNN state

**현재 상태**: ✅ Ve만 반환하도록 수정 완료

### 5.3 `update_inner` 메서드

**위치**: `efxplorer.py:363-442`

**구현 로직**:

```python
# 1. Compute Ve
bT_Ve, rnn_states_Ve, final_rnn_states_Ve = jax_vmap(
    ft.partial(self.scan_value,
               init_rnn_state_Ve=self.init_Vl_rnn_state,  # ← 수정 필요: init_value_rnn_state → init_Vl_rnn_state
               critic_params=critic_train_state.params)
)(rollout)

# 2. Compute Final Ve
def final_value_fn(graph, rnn_state):
    return self.critic.get_value(
        critic_train_state.params, 
        tree_index(graph, -1), 
        rnn_state  # ← 수정 필요: rnn_state_Ve → rnn_state
    )

final_Ve, _ = jax_vmap(final_value_fn)(rollout.next_graph, final_rnn_states_Ve)
bTp1_Ve = jnp.concatenate([bT_Ve, final_Ve[:, None]], axis=1)  # (b, T+1)

# 3. Team intrinsic reward
bT_Delta_team_int = rollout.intrinsic_rewards.sum(axis=-1)  # (b, T)

# 4. Compute extrinsic advantages Ae
def compute_Aext_single(rewards, values, next_values, dones):
    return compute_extrinsic_gae(
        values=values,
        rewards=rewards,
        next_values=next_values,
        dones=dones,
        gamma=self.gamma,
        gae_lambda=self.gae_lambda
    )

bTah_Aext = jax.vmap(compute_Aext_single)(
    rollout.rewards,      # (b, T, a)
    bT_Ve,               # (b, T)
    bTp1_Ve[:, 1:],      # (b, T)
    rollout.dones        # (b, T)
)  # (b, T, a)

# 5. Compute final advantages A_t^{(T)}
def compute_A_final_single(Aext, Delta_team_int, z):
    return compute_conservative_exploration_gae(
        Tah_Aext=Aext,                    # (T, a)
        T_Delta_team_int=Delta_team_int,  # (T,)
        T_z=z.squeeze(-1)                 # (T, a)
    )

bTah_A_final = jax.vmap(compute_A_final_single)(
    bTah_Aext,                     # (b, T, a)
    bT_Delta_team_int,             # (b, T)
    rollout.zs.squeeze(-1)         # (b, T, a)
)  # (b, T, a)

# 6. Normalize advantages
bTah_A_final = (bTah_A_final - bTah_A_final.mean(axis=1, keepdims=True)) / (
    bTah_A_final.std(axis=1, keepdims=True) + 1e-8
)

# 7. Update networks
def update_fn(carry, idx):
    critic, policy = carry
    rollout_batch = jtu.tree.map(lambda x: x[idx], rollout)
    critic, critic_info = self.update_critic(
        critic, rollout_batch, bT_Ve[idx], rnn_states_Ve[idx], rnn_chunk_ids
    )
    policy, policy_info = self.update_policy(
        policy, rollout_batch, bTah_A_final[idx], rnn_chunk_ids
    )
    return (critic, policy), (critic_info | policy_info)
```

**현재 상태**: ⚠️ 부분 완료 (버그 수정 필요)
- ✅ Conservative exploration 로직 구현됨
- ⚠️ Line 382: `init_value_rnn_state` → `init_Vl_rnn_state` 수정 필요
- ⚠️ Line 388: `rnn_state_Ve` → `rnn_state` 수정 필요
- ⚠️ Line 390-391: `final_Ve` shape 확인 필요

### 5.4 `update_critic` 메서드

**위치**: `efxplorer.py:444-472`

**현재 상태**: ✅ Ve만 업데이트 (정상)

### 5.5 `update_policy` 메서드

**위치**: `efxplorer.py:492-545`

**현재 상태**: ⚠️ 부분 완료
- ✅ Final advantage 사용
- ⚠️ Line 495-497: 불필요한 broadcast 제거 필요 (이미 agent-wise)

**수정 필요**:
```python
# 기존 (불필요한 broadcast)
gaes = gaes[:, :, None]
gaes = jnp.repeat(gaes, self.n_agents, axis=-1)

# 수정 후 (이미 agent-wise이므로 broadcast 불필요)
assert gaes.shape == (rollout.actions.shape[0], rollout.actions.shape[1], self.n_agents), \
    f"Expected gaes shape {(rollout.actions.shape[0], rollout.actions.shape[1], self.n_agents)}, got {gaes.shape}"
```

### 5.6 `update` 메서드

**위치**: `efxplorer.py:294-319`

**현재 상태**: ✅ Ve만 처리 (정상)

### 5.7 `save`/`load` 메서드

**위치**: `efxplorer.py:569-582`

**현재 상태**: ✅ Ve만 저장/로드 (정상)

---

## 6. 발견된 버그 및 수정 사항

### 6.1 `update_inner` 메서드 버그

**Line 382**: `init_value_rnn_state` → `init_Vl_rnn_state`
```python
# 수정 전
init_rnn_state=self.init_value_rnn_state,

# 수정 후
init_rnn_state_Ve=self.init_Vl_rnn_state,
```

**Line 388**: `rnn_state_Ve` → `rnn_state`
```python
# 수정 전
def final_value_fn(graph, rnn_state):
    return self.critic.get_value(critic_train_state.params, tree_index(graph, -1), rnn_state_Ve)

# 수정 후
def final_value_fn(graph, rnn_state):
    return self.critic.get_value(critic_train_state.params, tree_index(graph, -1), rnn_state)
```

**Line 390-391**: `final_Ve` shape 확인
```python
# 현재 코드
final_Ve, _ = jax_vmap(final_value_fn)(rollout.next_graph, final_rnn_states_Ve)  # (b, a, 1)
bTp1_Ve = jnp.concatenate([bT_Ve, final_Ve[:, None]], axis=1)  # (b, T, a, 1)

# 수정 필요: final_Ve가 (b,) shape이어야 함
# bT_Ve: (b, T)
# final_Ve: (b,) → squeeze 필요
final_Ve = final_Ve.squeeze()  # (b,)
bTp1_Ve = jnp.concatenate([bT_Ve, final_Ve[:, None]], axis=1)  # (b, T+1)
```

### 6.2 `update_policy` 메서드 수정

**Line 495-497**: 불필요한 broadcast 제거
```python
# 수정 전
gaes = gaes[:, :, None]
gaes = jnp.repeat(gaes, self.n_agents, axis=-1)

# 수정 후
# Final advantages are already agent-wise (b, T, a)
assert gaes.shape == (rollout.actions.shape[0], rollout.actions.shape[1], self.n_agents), \
    f"Expected gaes shape {(rollout.actions.shape[0], rollout.actions.shape[1], self.n_agents)}, got {gaes.shape}"
```

---

## 7. 완성도 체크리스트

### 7.1 Utils 함수 (`defmarl/algo/utils.py`)

- [x] `compute_conservative_exploration_gae` 구현 완료
- [x] `compute_extrinsic_gae` 구현 완료
- [x] Intrinsic GAE/Vi 관련 함수 없음 (원칙 준수)

**완성도**: ✅ **100%**

### 7.2 Rollout 함수 (`defmarl/trainer/utils.py`)

- [x] `rollout_efxplorer`: `intrinsic_reward = -log_pi * intrinsic_coef` (per-agent)
- [x] `Delta_team_int = intrinsic_reward.sum()` (team sum)
- [x] `z_next = clip(z - Delta_team_int, -env.reward_max, -env.reward_min)` (shared z)

**완성도**: ✅ **100%**

### 7.3 알고리즘 클래스 (`defmarl/algo/efxplorer.py`)

- [x] Vi 네트워크/하이퍼파라미터/저장·로드 완전 제거
- [x] `scan_value`: Ve만 반환
- [x] `update_inner`: Ve→Ae 계산, Δ_team_int·z로 `compute_conservative_exploration_gae` 호출
- [x] `update_critic`: Ve만 처리
- [x] `update`: Ve만 처리
- [x] `save`/`load`: Ve만 처리
- [ ] `update_inner`: 버그 수정 필요 (line 382, 388, 390-391)
- [ ] `update_policy`: 불필요한 broadcast 제거 필요 (line 495-497)

**완성도**: ⚠️ **85%** (버그 수정 필요)

### 7.4 Shape 검증

- [ ] `bT_Ve`: `(b, T)` 확인
- [ ] `bTah_Aext`: `(b, T, a)` 확인
- [ ] `bT_Delta_team_int`: `(b, T)` 확인
- [ ] `bTah_A_final`: `(b, T, a)` 확인
- [ ] z-dynamics가 Δ_team_int 기반으로 감소하는지 확인

**완성도**: ⚠️ **0%** (테스트 필요)

---

## 8. 수정 필요 사항 요약

### 우선순위 높음 (버그 수정)

1. **`update_inner` Line 382**: `init_value_rnn_state` → `init_Vl_rnn_state`
2. **`update_inner` Line 388**: `rnn_state_Ve` → `rnn_state`
3. **`update_inner` Line 390-391**: `final_Ve` shape 수정
4. **`update_policy` Line 495-497**: 불필요한 broadcast 제거

### 우선순위 중간 (검증)

5. Shape 검증 코드 추가
6. z-dynamics 수치 검증

---

## 9. 참조 파일

- **DefMARL 구조**: `defmarl/algo/defmarl.py`
- **Utils 함수**: `defmarl/algo/utils.py`
- **Rollout 함수**: `defmarl/trainer/utils.py`
- **체크리스트**: `IMPLEMENTATION_CHECKLIST.md`

---

## 10. 다음 단계

1. **버그 수정**: `update_inner`와 `update_policy`의 버그 수정
2. **Shape 검증**: 각 단계에서 shape assertion 추가
3. **수치 검증**: z-dynamics와 final advantage 계산 검증
4. **통합 테스트**: 전체 파이프라인 테스트

