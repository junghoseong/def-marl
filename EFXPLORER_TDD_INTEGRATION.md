# EFXplorer + TDD Intrinsic Integration 가이드

TDD(Trajectory Distance / MRN 기반) intrinsic을 EFXplorer 학습 루프에 넣는 절차를 정리한다. 목표: rollout으로 (s_t, s_{t+1}) 쌍을 모아 contrastive 학습 → TDD intrinsic을 `rollout.intrinsic_rewards`에 태깅 → 기존 PPO/Conservative loop를 그대로 사용.

---

## 1. 모듈 구조 (JAX/Flax)
- `intrinsic_reward.py`에 JAX/Flax TDD 모듈 추가:
  - `state_encoder`: MLP, φ(s)
  - `potential_net`: MLP, c(s')
  - `mrn_distance(φ_s, φ_sp)`: max + L2 조합
  - `tdd_logits(φ_s, φ_sp, c_sp) = c_sp^T - mrn_distance`
  - `tdd_loss`: InfoNCE(or variant)
- TrainState: `self.tdd_train_state` (params, optax optimizer).

## 2. Rollout에서 intrinsic 생성
- `trainer/utils.py: rollout_efxplorer`에서 각 step에 TDD intrinsic 계산:
  1) `(obs_t, obs_tp1)` 준비 (필요하면 flatten/feature 추출).
  2) `intrinsic_reward_tdd = tdd_intrinsic_fn(tdd_params, obs_t, obs_tp1)`  (per-agent로 동일 태깅하거나, 에이전트 별 특징이면 그에 맞게 확장).
  3) `rollout.intrinsic_rewards`를 TDD 결과로 채움 (shape `(b, T, a)`).
  4) 팀 합 `Delta_team_int = intrinsic_reward.sum(axis=-1)`로 z-dynamics/DP에 사용.
- 참고: 기존 entropy-based intrinsic을 대체하거나, 합성할 경우 가중합으로 합치고 shape 유지.

## 3. TDD 학습 (update_intrinsic)
- `efxplorer.py`에 `update_intrinsic(self, rollout)` 추가:
  - 배치 구성: `(obs_t, obs_tp1)` → reshape `(b*T*a, obs_dim)` 또는 샘플링.
  - `tdd_loss, logs = tdd_loss_fn(tdd_params, batch_obs_t, batch_obs_tp1)`
  - `grads = jax.grad(loss_fn)(tdd_params)` → `tdd_train_state.apply_gradients`.
  - 로그: `tdd/loss`, `tdd/logits_pos`, `tdd/logits_neg`, `tdd/logits_logsumexp`, `tdd/acc`.
- 호출 시점: `update`에서 PPO 이전 또는 이후에 1 step 호출 (예: `update_intrinsic` → `update_inner` → `update_z`).

## 4. Trainer/Buffer 수정
- Trainer:
  - `collect`는 기존대로 `rollout_efxplorer` 호출(내부에서 intrinsic 채움).
  - `update` 호출 전후로 `update_intrinsic` 한 번 실행.
  - `get_opt_z` 그대로 사용(평가 시 z=0).
- Buffer:
  - Rollout 구조(`Rollout.intrinsic_rewards`)는 이미 존재. TDD intrinsic으로 덮어씀.
  - 추가로 obs_t/obs_tp1가 필요하면 rollout에 포함된 `graph`/`next_graph`에서 feature 추출하여 TDD 학습 배치 생성.

## 5. 변수/차원 기준 (최신 EFXplorer 흐름)
- Rollout 출력:
  - `rewards`: (b, T, a)
  - `intrinsic_rewards`: (b, T, a) ← TDD 결과
  - `zs`: (b, T, a, 1)
  - `dones`: (b, T)
  - `graph`, `next_graph`: obs/feature 추출용
- Update_inner 입력/출력:
  - `bT_Ve`: (b, T, 1), `bTp1_Ve`: (b, T+1, 1)
  - `bT_Delta_team_int`: (b, T) = intrinsic_rewards.sum(-1)
  - `Ae`: (b, T, a) via `compute_gae_single` per env
  - `A_final`: (b, T, a) via `compute_conservative_exploration_gae(Ae, Δ_team_int, z)`
- Value loss:
  - targets (b, T, 1) → squeeze/reshape (b, *)
  - values (b, n_chunks, T_chunk) → reshape (b, *)

## 6. 통합 순서 예시 (efxplorer.py)
1) `__init__`: TDD 모듈/optimizer/TrainState 초기화 (`self.tdd_train_state`)
2) `rollout_fn`: `rollout_efxplorer`에서 TDD intrinsic 계산 후 `intrinsic_rewards` 채움
3) `update`:
   - `self.tdd_train_state = update_intrinsic(rollout, self.tdd_train_state)`
   - `critic, policy = update_inner(...)`
   - `self.z_current = update_z(rollout, z_old)`
4) `save/load`: TDD params 포함

## 7. 실험/검증 체크
- 로그: `tdd/loss`, `tdd/logits_*`, `tdd/acc`, `intrinsic/mean`, `z/mean`, `policy/critic loss`.
- Shape 확인: rollout 전/후 `(b,T,a)` 유지, TDD 학습 배치 `(b*T*a, obs_dim)` 또는 샘플링.
- 성능: TDD intrinsic이 학습/반영되고 z-dynamics가 기대대로 감소하는지 모니터링.

