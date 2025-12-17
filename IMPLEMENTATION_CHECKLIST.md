# Conservative Exploration Framework 체크리스트 (최신)

## 원칙
- Vi/Intrinsic value는 사용하지 않는다. Ve/Ae, Δ_team_int, z만 사용.
- 용어: Ve = extrinsic value, Ae = extrinsic advantage.

## 1. `defmarl/algo/utils.py`
- [x] `compute_conservative_exploration_gae(Ae, T_Delta_team_int, T_z)` 구현 완료 ✅
- [x] `compute_extrinsic_gae` 구현 완료 ✅
- [x] Intrinsic GAE/Vi 관련 함수 없음 (원칙 준수) ✅

**완성도**: ✅ **100%**

## 2. `defmarl/trainer/utils.py`
- [x] `rollout_efxplorer`: `intrinsic_reward = -log_pi * intrinsic_coef` (per-agent) ✅
- [x] `Delta_team_int = intrinsic_reward.sum()` (team sum) ✅
- [x] `z_next = clip(z - Delta_team_int, -env.reward_max, -env.reward_min)` (shared z) ✅

**완성도**: ✅ **100%**

## 3. `defmarl/algo/efxplorer.py`
- [x] Vi 네트워크/하이퍼파라미터/저장·로드 완전 제거 ✅
- [x] `scan_value`: Ve만 반환 ✅
- [x] `update_inner`: Ve→Ae 계산, Δ_team_int·z로 `compute_conservative_exploration_gae(Ae, Δ_team_int, z)` 호출 ✅
- [x] `update_critic`: Ve만 처리 ✅
- [x] `update`: Ve만 처리 ✅
- [x] `save`/`load`: Ve만 처리 ✅
- [ ] **버그 수정 필요**: `update_inner` Line 382: `init_value_rnn_state` → `init_Vl_rnn_state` ⚠️
- [ ] **버그 수정 필요**: `update_inner` Line 388: `rnn_state_Ve` → `rnn_state` ⚠️
- [ ] **버그 수정 필요**: `update_inner` Line 390-391: `final_Ve` shape 수정 ⚠️
- [ ] **버그 수정 필요**: `update_policy` Line 495-497: 불필요한 broadcast 제거 ⚠️

**완성도**: ⚠️ **85%** (버그 수정 필요)

## 4. 테스트 체크
- [ ] Shape 검증: `bT_Ve (b,T)`, `bTah_Ae (b,T,a)`, `bT_Delta_team_int (b,T)`, `bTah_A_final (b,T,a)`
- [ ] z-dynamics가 Δ_team_int 기반으로 감소하는지 확인
- [ ] Final advantage 계산 정확성 검증

**완성도**: ⚠️ **0%** (테스트 필요)

---

## 발견된 버그 상세

### 버그 1: `update_inner` Line 382
```python
# 현재 (에러 발생 가능)
init_rnn_state=self.init_value_rnn_state,

# 수정 필요
init_rnn_state_Ve=self.init_Vl_rnn_state,
```

### 버그 2: `update_inner` Line 388
```python
# 현재 (에러 발생)
def final_value_fn(graph, rnn_state):
    return self.critic.get_value(critic_train_state.params, tree_index(graph, -1), rnn_state_Ve)

# 수정 필요
def final_value_fn(graph, rnn_state):
    return self.critic.get_value(critic_train_state.params, tree_index(graph, -1), rnn_state)
```

### 버그 3: `update_inner` Line 390-391
```python
# 현재 (shape 불일치 가능)
final_Ve, _ = jax_vmap(final_value_fn)(rollout.next_graph, final_rnn_states_Ve)  # (b, a, 1)
bTp1_Ve = jnp.concatenate([bT_Ve, final_Ve[:, None]], axis=1)  # (b, T, a, 1)

# 수정 필요
final_Ve, _ = jax_vmap(final_value_fn)(rollout.next_graph, final_rnn_states_Ve)
final_Ve = final_Ve.squeeze()  # (b,)
bTp1_Ve = jnp.concatenate([bT_Ve, final_Ve[:, None]], axis=1)  # (b, T+1)
```

### 버그 4: `update_policy` Line 495-497
```python
# 현재 (불필요한 broadcast)
gaes = gaes[:, :, None]
gaes = jnp.repeat(gaes, self.n_agents, axis=-1)

# 수정 필요 (이미 agent-wise)
assert gaes.shape == (rollout.actions.shape[0], rollout.actions.shape[1], self.n_agents)
```

---

## 전체 완성도 요약

| 항목 | 완성도 | 상태 |
|------|--------|------|
| Utils 함수 | 100% | ✅ 완료 |
| Rollout 함수 | 100% | ✅ 완료 |
| 알고리즘 클래스 | 85% | ⚠️ 버그 수정 필요 |
| 테스트/검증 | 0% | ⚠️ 테스트 필요 |
| **전체** | **71%** | ⚠️ **진행 중** |

---

## 다음 단계

1. **즉시 수정**: 4개 버그 수정
2. **Shape 검증**: 각 단계에서 assertion 추가
3. **수치 검증**: z-dynamics 및 final advantage 검증
4. **통합 테스트**: 전체 파이프라인 테스트




