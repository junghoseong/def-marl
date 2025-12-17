# EFXplorer 학습 루프 분석 및 개선 사항

**최종 업데이트**: 학습 가능 상태 확인 완료 ✅

## 1. 현재 구현 상태 요약

### 1.1 EFXplorer의 핵심 특징
- **Vi (Intrinsic Value) 네트워크 제거**: Ve (Extrinsic Value)만 사용
- **Conservative Exploration Framework**: 
  - Final Advantage: `A_t^{(T)} = min{A_t^ext, A_{t+1}^ext, ..., A_T^ext, S_t}`
  - S_t = R_t^int - z_t (intrinsic return - budget)
- **z-dynamics**: `z_{t+1} = z_t - Δ_t^team-int` (rollout 중 자동 업데이트)
- **Team-level intrinsic reward**: `Δ_t^team-int = Σ_i r_{i,t}^int = Σ_i (-log π_i * intrinsic_coef)`
- **z 업데이트 전략**: 이전 rollout의 intrinsic reward 합의 평균으로 설정 (EMA 방식은 주석처리)
- **초기 z**: 0으로 초기화
- **평가 시 z**: 0으로 설정 (conditioning)

### 1.2 DefMARL과의 차이점
| 항목 | DefMARL | EFXplorer |
|------|---------|-----------|
| Value Networks | Vl (extrinsic), Vh (constraint) | Ve (extrinsic)만 |
| z 업데이트 | Root finder로 optimal z 계산 | 이전 rollout의 intrinsic reward 합 |
| z 초기화 | `get_opt_z`로 계산 | 0으로 초기화 |
| Test 시 z 선택 | `get_opt_z` 사용 | z=0 사용 |

---

## 2. 학습 루프 구조 분석

### 2.1 현재 학습 루프 (trainer.py)
```python
for step in range(0, self.steps + 1):
    # 1. 평가 (eval_interval마다)
    if step % self.eval_interval == 0:
        # test_opt_fn은 get_opt_z를 사용 (EFXplorer에는 없음)
        test_rollouts = test_opt_fn(self.algo.params, test_keys)
    
    # 2. 데이터 수집
    rollouts = self.algo.collect(self.algo.params, key_x0)
    
    # 3. 알고리즘 업데이트
    update_info = self.algo.update(rollouts, step)
```

### 2.2 EFXplorer의 update 메서드 흐름
```python
def update(self, rollout: Rollout, step: int) -> dict:
    # 1. PPO epoch 반복
    for i_epoch in range(self.epoch_ppo):
        # 2. update_inner 호출
        critic_train_state, policy_train_state, update_info = self.update_inner(...)
    
    # ❌ z 업데이트 없음!
    return update_info
```

---

## 3. 발견된 문제점 및 미흡한 부분

### 3.1 ✅ **완료: z 업데이트 구현**

**구현 상태**: ✅ 완료

**구현 내용**:
- `update_z` 메서드: 이전 rollout의 intrinsic reward 합의 평균으로 z 설정
- `update` 메서드에서 z 업데이트 로직 추가
- 초기 z = 0으로 설정 (`__init__`에서 `self.z_current = jnp.zeros((self.n_agents, 1))`)

**현재 코드** (`efxplorer.py:561-591`):
```python
def update_z(self, rollout: Rollout, z_old: Float[Array, "a 1"]) -> Float[Array, "a 1"]:
    """
    Update budget z based on previous rollout's intrinsic reward sum
    
    z_new = sum of intrinsic rewards from previous rollout
    """
    # Compute total intrinsic reward per trajectory
    intrinsic_return_per_traj = rollout.intrinsic_rewards.sum(axis=(1, 2))  # (b,)
    intrinsic_return_mean = intrinsic_return_per_traj.mean()
    
    # Set z to the mean intrinsic return (shared across all agents)
    z_new = jnp.full((self.n_agents, 1), intrinsic_return_mean)
    z_new = jnp.clip(z_new, -self._env.reward_max, -self._env.reward_min)
    
    # EMA approach (commented out - design choice)
    # lr_z = 1e-3
    # z_update = lr_z * intrinsic_return_mean
    # z_new = z_old + z_update
    
    return z_new
```

**update 메서드** (`efxplorer.py:321-328`):
```python
# Update z based on previous rollout's intrinsic reward sum
z_old = self.z_current
self.z_current = self.update_z(rollout, z_old)

# Add z update info to logging
update_info['z/mean'] = self.z_current.mean()
update_info['z/change'] = (self.z_current - z_old).mean()
update_info['z/value'] = self.z_current[0, 0]
```

---

### 3.2 ✅ **완료: get_opt_z 메서드 구현**

**구현 상태**: ✅ 완료

**구현 내용**:
- 평가 시 z=0을 반환하도록 구현
- `trainer.py`의 `hasattr` 체크와 호환

**현재 코드** (`efxplorer.py:600-615`):
```python
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
```

---

### 3.3 ✅ **완료: update_policy의 shape assertion 수정**

**구현 상태**: ✅ 완료

**수정 내용**:
- `gaes.shape` assertion을 `(b, T, n_agents)`로 수정

**현재 코드** (`efxplorer.py:507-509`):
```python
# Final advantages are agent-wise (b, T, a)
assert gaes.shape == (rollout.actions.shape[0], rollout.actions.shape[1], self.n_agents), \
    f"Expected gaes shape {(rollout.actions.shape[0], rollout.actions.shape[1], self.n_agents)}, got {gaes.shape}"
```

---

### 3.4 🟡 **우선순위 중간: z 초기화 전략**

**문제**: 
- Rollout 시 z가 랜덤하게 초기화됨 (`rollout_efxplorer:106-111`)
- 학습된 z를 활용하지 않음

**현재 코드** (`trainer/utils.py:106-111`):
```python
z0 = jax.random.uniform(key_z0, (1, 1), minval=-env.reward_max, maxval=-env.reward_min)
rng = jax.random.uniform(z_key, (1, 1))
z0 = jnp.where(rng > 0.7, -env.reward_max, z0)
z0 = jnp.where(rng < 0.2, -env.reward_min, z0)
```

**개선 방안**:
- 학습 초기에는 랜덤, 이후에는 학습된 z 사용
- 또는 z를 state로 관리하여 점진적으로 업데이트

---

### 3.5 ✅ **완료: update_inner의 수정 사항**

**구현 상태**: ✅ 완료

**수정 내용**:
1. `scan_value` 호출 시 `rollout.graph` 대신 `rollout` 전체 전달
2. `final_value_fn`에서 z 전달 추가
3. `final_Ve` shape 처리 (squeeze 추가)

**현재 코드** (`efxplorer.py:392-405`):
```python
# 1. Compute Ve
bT_Ve, rnn_states_Ve, final_rnn_states_Ve = jax_vmap(
    ft.partial(self.scan_value,
               init_rnn_state_Ve=self.init_Vl_rnn_state,
               critic_params=critic_train_state.params)
)(rollout)  # values: (b, T)

# 2. Compute Final Ve
def final_value_fn(graph, zs, rnn_state):
    # Use the last z value from the trajectory
    z_final = zs[-1][0][None, :] if zs is not None else None
    return self.critic.get_value(critic_train_state.params, tree_index(graph, -1), rnn_state, z_final)

final_Ve, _ = jax.vmap(final_value_fn)(rollout.next_graph, rollout.zs, final_rnn_states_Ve)
final_Ve = final_Ve.squeeze()
bTp1_Ve = jnp.concatenate([bT_Ve, final_Ve[:, None]], axis=1) # (b, T+1)
```

### 3.6 ✅ **완료: update_critic의 수정 사항**

**구현 상태**: ✅ 완료

**수정 내용**:
- `update_critic`에서 chunk로 나눈 rollout 전체를 전달하도록 수정

**현재 코드** (`efxplorer.py:468-478`):
```python
# Create chunked rollout
bcT_rollout = jax.tree.map(lambda x: x[:, rnn_chunk_ids], rollout)
rnn_state_inits = jnp.zeros_like(rnn_states[:, rnn_chunk_ids[:, 0]])

def get_value_loss(params):
    values, value_rnn_states, final_value_rnn_states = jax.vmap(jax.vmap(
        ft.partial(self.scan_value,
                   init_rnn_state_Ve=rnn_state_inits,
                   critic_params=params)
    ))(bcT_rollout)  # values: (b, n_chunks, T_chunk)
    values = values.reshape((values.shape[0], -1))
    loss_critic = optax.l2_loss(values, targets).mean()
    return loss_critic
```

---

### 3.7 🟢 **우선순위 낮음: z-dynamics 검증**

**상태**: 학습 중 검증 가능

**검증 방법**:
```python
# update_inner에서 디버깅 정보 추가
z_diff = rollout.zs[:, 1:] - rollout.zs[:, :-1]  # (b, T-1, a, 1)
Delta_team_int_expected = -rollout.intrinsic_rewards.sum(axis=-1)[:, :-1]  # (b, T-1)
# z_diff와 Delta_team_int_expected 비교
```

---

## 4. 학습 루프 개선 제안

### 4.1 수정된 update 메서드

```python
def update(self, rollout: Rollout, step: int) -> dict:
    key, self.key = jr.split(self.key)
    
    update_info = {}
    assert rollout.dones.shape[0] * rollout.dones.shape[1] >= self.batch_size
    
    # PPO 업데이트
    for i_epoch in range(self.epoch_ppo):
        idx = np.arange(rollout.dones.shape[0])
        np.random.shuffle(idx)
        rnn_chunk_ids = jnp.arange(rollout.dones.shape[1])
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
    
    # ✅ 추가: z 업데이트 (Outer Loop)
    if not hasattr(self, 'z_current'):
        # 초기 z: rollout의 평균 z 사용
        self.z_current = rollout.zs.mean(axis=(0, 1))  # (a, 1)
    
    # z 업데이트 (논문의 Outer Loop)
    z_old = self.z_current
    self.z_current = self.update_z(rollout, z_old)
    
    # z 업데이트 정보 추가
    update_info['z/mean'] = self.z_current.mean()
    update_info['z/change'] = (self.z_current - z_old).mean()
    
    if self.use_prev_init:
        self.memory = rollout
    
    return update_info
```

### 4.2 추가할 get_opt_z 메서드

```python
def get_opt_z(
    self,
    graph: GraphsTuple,
    rnn_state: Array,
    params: Optional[Params] = None
) -> Tuple[Array, Array]:
    """
    평가 시 사용할 z 반환
    EFXplorer의 경우 학습된 z_current 사용
    """
    if hasattr(self, 'z_current'):
        z = self.z_current
    else:
        # 기본값: 중간값
        z_mid = (-self._env.reward_max - self._env.reward_min) / 2
        z = jnp.array([[z_mid]]).repeat(self.n_agents, axis=0)
    
    return z, rnn_state
```

---

## 5. 체크리스트

### 5.1 필수 수정 사항 ✅
- [x] `update` 메서드에 z 업데이트 로직 추가 ✅
- [x] `get_opt_z` 메서드 구현 ✅
- [x] `update_policy`의 shape assertion 수정 ✅
- [x] `update_inner`에서 `scan_value` 호출 수정 (rollout 전체 전달) ✅
- [x] `final_value_fn`에서 z 전달 추가 ✅
- [x] `update_critic`에서 chunk rollout 전달 수정 ✅

### 5.2 학습 가능성 확인 ✅
- [x] `make_algo`에서 `efxplorer` 등록 확인 ✅
- [x] `train.py`에서 필요한 하이퍼파라미터 전달 확인 ✅
- [x] `intrinsic_coef` 기본값 설정 확인 (1.0) ✅
- [x] 불필요한 인자 (`Vh_gnn_layers`, `lagr_init`, `lr_lagr`)는 `**kwargs`로 처리 ✅

### 5.3 권장 개선 사항 (선택적)
- [ ] z-dynamics 검증 코드 추가 (학습 중 검증 가능)
- [ ] z 업데이트 로깅 강화 (현재 기본 로깅 구현됨)
- [ ] z 학습률 스케줄링 (현재는 intrinsic reward sum 직접 사용)

---

## 6. 참고: 데이터 흐름 & 텐서 차원 (rollout→update)

### 6.1 Rollout (trainer/utils.py)
- `rollout_efxplorer` 출력
  - `rewards`: (b, T, a) 또는 (b, T) → 사용 전 (b, T, a)로 확장
  - `intrinsic_rewards`: (b, T, a), per-agent `-log_pi * intrinsic_coef`
  - `zs`: (b, T, a, 1), shared z but tiled per agent
  - `dones`: (b, T)

### 6.2 update_inner (efxplorer.py)
- 입력 shapes:
  - `bT_Ve`: (b, T, 1), `bTp1_Ve`: (b, T+1, 1)
  - `rollout.rewards`: (b, T, a)로 보정하여 사용
  - `bT_Delta_team_int`: (b, T) = intrinsic_rewards.sum(axis=-1)
- Ae 계산 (per env):
  - rewards_sum = rewards.sum(-1) → (T,)
  - `compute_gae_single(values.squeeze(-1), rewards_sum, dones, next_values.squeeze(-1))`
  - Ae: (T, a) via repeat over agents
- A_final:
  - `compute_conservative_exploration_gae(Ae (T,a), Δ_team_int (T,), z (T,a))`
  - 출력: (T, a), 배치 vmap 후 (b, T, a)
- 정규화:
  - `(b, T, a)` 단위로 평균/표준편차

### 6.3 update_critic (efxplorer.py)
- 입력:
  - `targets`: (b, T, 1) → squeeze → (b, T) → reshape (b, *)
  - `values`: (b, n_chunks, T_chunk) → reshape (b, *)
- 손실:
  - `optax.l2_loss(values, targets)`; shape 정렬 이후 실행

### 6.4 z 업데이트
- `update_z`:
  - intrinsic_rewards.sum(axis=(1,2)) → (b,)
  - mean → scalar → z_new = fill((a,1), mean), clip to [-reward_max, -reward_min]
- `get_opt_z`: 평가 시 z=0 반환 (shape (a,1))

### 6.5 체크용 shape 리스트
- `rollout.rewards`: 기대 (b, T, a)
- `rollout.intrinsic_rewards`: (b, T, a)
- `rollout.zs`: (b, T, a, 1)
- `bT_Ve`: (b, T, 1), `bTp1_Ve`: (b, T+1, 1)
- `bT_Delta_team_int`: (b, T)
- `Ae (b, T, a)`, `A_final (b, T, a)`

---

## 7. 학습 실행
```bash
python train.py --env MPETarget --algo efxplorer -n 3 --obs 3
```

검증 시 확인:
- `z/mean`, `z/change`, `z/value`
- `policy/loss`, `critic/loss` 수렴
- 필요시 rollout 중 z-dynamics 모니터링

