# DefMARL vs EFXplorer: max vs min 분석

## 1. 핵심 차이점 요약

### DefMARL: `max` 사용 (Epigraph Form)

**목적**: Constraint 만족 + Reward 최대화

**핵심 함수**: `compute_dec_efocp_V` (utils.py:118-125)
```python
def compute_dec_efocp_V(z, Vhs, Vl):
    # V_i(x^τ, z; π) = max{V_i^h(o_i^τ; π), V^l(x^τ; π) - z}
    return jnp.maximum(Vhs.max(-1), (Vl - z))
```

**의미**:
- `Vh`: Constraint value (cost 예측)
- `Vl - z`: Reward value (reward - budget)
- **max**: 둘 중 더 큰 값을 선택 → 둘 다 만족해야 함
- Epigraph form의 핵심: Constraint 위반을 피하면서 reward를 최대화

**사용 위치**: `defmarl.py:457`
```python
bTa_V = jax_vmap(jax_vmap(compute_dec_efocp_V))(
    rollout.zs.squeeze(-1)[:, :, 0], bTah_Vh, bT_Vl
)
bTa_A = bTa_Q - bTa_V  # Advantage 계산
```

---

### EFXplorer: `min` 사용 (Conservative Exploration)

**목적**: Exploration vs Exploitation 균형

**핵심 함수**: `compute_conservative_exploration_gae` (utils.py:126-199)
```python
def compute_conservative_exploration_gae(
    Tah_Aext,      # Extrinsic advantages
    T_Delta_team_int,  # Team intrinsic rewards
    T_z            # Budget variables
):
    # A_t^{(T)} = min{A_t^ext, A_{t+1}^ext, ..., A_T^ext, S_t}
    # where S_t = R_t^int - z_t
```

**의미**:
- `A_ext`: Extrinsic advantage (task performance)
- `S_t = R_t^int - z_t`: Intrinsic reward - budget
- **min**: 둘 중 더 작은 값을 선택 → 보수적 접근
- Conservative exploration: Exploration과 exploitation 중 더 보수적인 선택

**사용 위치**: `efxplorer.py:440-448`
```python
bTah_A_final = jax.vmap(compute_A_final_single)(
    bTah_Aext, bT_Delta_team_int, rollout.zs.squeeze(-1)
)
```

---

## 2. 알고리즘적 차이점

### DefMARL의 Value 계산 흐름

```
1. Vh 계산 (constraint value): Vh(o_i, z) - 각 constraint의 미래 cost 예측
2. Vl 계산 (reward value): Vl(x, z) - 미래 reward 예측
3. V = max{Vh, Vl - z} - 둘 중 더 큰 값 선택
4. Q 계산: GAE를 통해 Q-value 계산
5. A = Q - V: Advantage 계산
```

**특징**:
- Constraint와 reward를 **분리**하여 학습
- `max`로 인해 둘 다 만족해야 함
- z는 budget 역할 (constraint와 reward 사이 trade-off)

---

### EFXplorer의 Advantage 계산 흐름

```
1. Ve 계산 (extrinsic value): Global extrinsic value
2. A_ext 계산: Extrinsic advantage (GAE)
3. S_t 계산: R_t^int - z_t (intrinsic reward - budget)
4. A_final = min{A_ext, A_{t+1}^ext, ..., A_T^ext, S_t} - backward DP
```

**특징**:
- Extrinsic과 intrinsic을 **결합**하여 학습
- `min`으로 인해 보수적 선택
- z는 exploration budget 역할

---

## 3. L.187 수정의 문제점

### 현재 수정 (L.187)
```python
# 원래 코드 (L.186):
A_T_final = jnp.minimum(A_T_ext, S_T)

# 수정 후 (L.187):
A_T_final = A_T_ext
```

### 문제점

#### 1. **일관성 부족**
- Terminal step (T)에서만 intrinsic 무시
- Backward scan (L.180)에서는 여전히 `min` 사용:
  ```python
  A_t = jnp.minimum(A_ext_row, A_next)  # ← 여전히 min 사용
  ```
- 결과: Terminal에서는 extrinsic만, 나머지에서는 min 사용 → 일관성 없음

#### 2. **알고리즘 의도와 불일치**
- Conservative exploration의 핵심: `min`으로 보수적 선택
- Terminal에서만 extrinsic 사용하면 전체 알고리즘의 의도와 맞지 않음

#### 3. **학습 불안정성**
- Terminal과 중간 step의 계산 방식이 다름
- Backward propagation 시 일관성 없는 gradient

---

## 4. Extrinsic Reward만 사용하려면?

### 옵션 1: 전체 알고리즘 수정 (권장)

**`compute_conservative_exploration_gae`를 완전히 우회**:

```python
# efxplorer.py:439-448 수정
# 기존:
bTah_A_final = jax.vmap(compute_A_final_single)(
    bTah_Aext, bT_Delta_team_int, rollout.zs.squeeze(-1)
)

# 수정 후 (extrinsic만 사용):
bTah_A_final = bTah_Aext  # ← 직접 사용
```

**장점**:
- 일관성 있음
- DefMARL과 유사한 구조 (단순 GAE 사용)

**단점**:
- Conservative exploration 알고리즘을 완전히 우회
- Intrinsic reward 계산은 하지만 사용 안 함

---

### 옵션 2: z를 매우 크게 설정

**z를 매우 큰 값으로 설정**하여 `S_t = R_t^int - z_t`가 항상 음수가 되도록:

```python
# efxplorer.py:update_z 수정
def update_z(self, rollout, z_old):
    # z를 매우 크게 설정하여 S_t가 항상 음수가 되도록
    return jnp.ones_like(z_old) * 1e10  # 매우 큰 값
```

**장점**:
- 알고리즘 구조 유지
- `min` 연산에서 항상 `A_ext` 선택

**단점**:
- z의 의미가 없어짐
- Intrinsic reward 계산은 하지만 의미 없음

---

### 옵션 3: `compute_conservative_exploration_gae` 내부 수정

**L.180과 L.187 모두 수정**:

```python
def compute_conservative_exploration_gae(...):
    ...
    def compute_A_final(carry, inp):
        A_ext_row, S_row = inp
        A_next = carry
        # 수정: min 대신 A_ext만 사용
        A_t = A_ext_row  # ← min 제거
        return A_t, A_t
    
    # Terminal도 수정
    A_T_final = A_T_ext  # ← 이미 수정됨 (L.187)
```

**장점**:
- 일관성 있음
- 알고리즘 구조 유지

**단점**:
- Conservative exploration의 의미가 없어짐
- 함수 이름과 실제 동작이 불일치

---

## 5. DefMARL과 동일하게 학습하려면?

### DefMARL의 구조

1. **Value 계산**: `V = max{Vh, Vl - z}`
2. **GAE 계산**: `compute_dec_efocp_gae` 사용
3. **Advantage**: `A = Q - V`

### EFXplorer를 DefMARL처럼 수정

```python
# efxplorer.py:update_inner 수정

# 1. Ve 계산 (기존과 동일)
bT_Ve, rnn_states_Ve, final_rnn_states_Ve = ...

# 2. Final Ve 계산 (기존과 동일)
bTp1_Ve = ...

# 3. Extrinsic GAE 계산 (DefMARL처럼)
def compute_extrinsic_gae_single(rewards, values, next_values, dones):
    rewards_sum = rewards.sum(axis=-1)  # (T,)
    targets, gaes = compute_gae_single(
        values=values.squeeze(-1),
        rewards=rewards_sum,
        dones=dones,
        next_values=next_values.squeeze(-1),
        gamma=self.gamma,
        gae_lambda=self.gae_lambda
    )
    return gaes[:, None].repeat(rewards.shape[1], axis=1)  # (T, a)

bTah_Aext = jax.vmap(compute_extrinsic_gae_single)(
    rollout_rewards, bT_Ve, bTp1_Ve[:, 1:], rollout.dones
)

# 4. Intrinsic reward 무시하고 A_ext만 사용
bTah_A_final = bTah_Aext  # ← Conservative exploration 우회

# 5. Normalize
bTah_A_final = (bTah_A_final - bTah_A_final.mean(axis=1, keepdims=True)) / (
    bTah_A_final.std(axis=1, keepdims=True) + 1e-8
)
```

**이렇게 하면**:
- DefMARL과 동일한 구조 (단순 GAE 사용)
- Extrinsic reward만 사용
- 일관성 있음

---

## 6. 권장 수정 사항

### 현재 문제
- L.187에서 terminal만 수정 → 일관성 없음
- Backward scan에서 여전히 `min` 사용

### 권장 수정

**옵션 A: 전체 알고리즘 수정 (extrinsic만 사용)**
```python
# efxplorer.py:439-448
# Conservative exploration 완전히 우회
bTah_A_final = bTah_Aext  # 직접 사용
```

**옵션 B: `compute_conservative_exploration_gae` 내부 수정**
```python
# utils.py:175-180
def compute_A_final(carry, inp):
    A_ext_row, S_row = inp
    A_next = carry
    # min 제거, A_ext만 사용
    A_t = A_ext_row  # ← 수정
    return A_t, A_t
```

---

## 7. 학습이 안 되는 이유

1. **일관성 부족**: Terminal과 중간 step의 계산 방식이 다름
2. **Gradient 불안정**: Backward propagation 시 일관성 없는 gradient
3. **알고리즘 의도와 불일치**: Conservative exploration의 핵심인 `min`을 부분적으로만 제거

### 해결 방법

**DefMARL과 동일하게 extrinsic만 사용**하려면:
- `compute_conservative_exploration_gae` 완전히 우회
- 또는 함수 내부에서 `min` 제거
- 일관성 있게 수정

---

## 8. 요약

| 항목 | DefMARL | EFXplorer (원래) | EFXplorer (수정 후) |
|------|---------|------------------|---------------------|
| **연산** | `max{Vh, Vl-z}` | `min{A_ext, S}` | `A_ext` (부분적) |
| **목적** | Constraint 만족 | Exploration 균형 | Extrinsic만 |
| **일관성** | ✅ 일관성 있음 | ✅ 일관성 있음 | ❌ 일관성 없음 |
| **학습 안정성** | ✅ 안정적 | ✅ 안정적 | ❌ 불안정 |

**결론**: L.187만 수정하는 것은 부족합니다. 전체 알고리즘을 일관성 있게 수정해야 합니다.

