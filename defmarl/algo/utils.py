import functools as ft
import jax
import jax.numpy as jnp
import einops as ei

from typing import Tuple
from jaxtyping import Float

from ..utils.typing import Array, Done, Reward, TFloat, BFloat, FloatScalar
from ..utils.utils import assert_shape


def compute_gae_single(
        values: Array, rewards: Reward, dones: Done, next_values: Array, gamma: float, gae_lambda: float
) -> Tuple[Array, Array]:
    """
    Compute generalized advantage estimation.
    """
    deltas = rewards + gamma * next_values * (1 - dones) - values
    gaes = deltas

    def scan_fn(gae, inp):
        delta, done = inp
        gae_prev = delta + gamma * gae_lambda * (1 - done) * gae
        return gae_prev, gae_prev

    _, gaes_prev = jax.lax.scan(scan_fn, gaes[-1], (deltas[:-1], dones[:-1]), reverse=True)
    gaes = jnp.concatenate([gaes_prev, gaes[-1, None]], axis=0)

    return gaes + values, (gaes - gaes.mean()) / (gaes.std() + 1e-8)


def compute_gae(
        values: Array, rewards: Reward, dones: Done, next_values: Array, gamma: float, gae_lambda: float
) -> Tuple[Array, Array]:
    return jax.vmap(ft.partial(
        compute_gae_single, gamma=gamma, gae_lambda=gae_lambda))(values, rewards, dones, next_values)


def compute_dec_efocp_gae(
    Tah_hs: Float[Array, "T a nh"],
    T_l: TFloat,
    T_z: TFloat,
    Tp1ah_Vh: Float[Array, "Tp1 a nh"],
    Tp1_Vl: Float[Array, "Tp1"],
    disc_gamma: float,
    gae_lambda: float,
    discount_to_max: bool = True
) -> tuple[Float[Array, "T a nh"], TFloat, Float[Array, "T a"]]:
    """
    Compute GAE for stabilize-avoid. Compute it using DP, starting at V(x_T) and working backwards.

    Returns
    -------
    Qhs: (T, a, nh),
    Ql: (T, )
    Q: (T, a),
    """
    T, n_agent, nh = Tah_hs.shape

    def loop(carry, inp):
        ii, hs, l, z, Vhs, Vl = inp  # hs: (a, nh), Vhs: (a, nh)
        next_Vhs_row, next_Vl_row, gae_coeffs = carry

        mask = assert_shape(jnp.arange(T + 1) < ii + 1, T + 1)
        mask_l = assert_shape(mask[:, None], (T + 1, 1))
        mask_h = assert_shape(mask[:, None, None], (T + 1, 1, 1))

        # DP for Vh.
        if discount_to_max:
            h_disc = hs.max(-1)  # (a,)
        else:
            h_disc = hs

        disc_to_h = (1 - disc_gamma) * h_disc[None, :, None] + disc_gamma * next_Vhs_row  # (T + 1, a, h)
        Vhs_row = assert_shape(mask_h * jnp.maximum(hs, disc_to_h), (T + 1, n_agent, nh), "Vhs_row")
        # DP for Vl. Clamp it to within J_max so it doesn't get out of hand.
        Vl_row = assert_shape(mask_l * (l + disc_gamma * next_Vl_row), (T + 1, n_agent))
        # Vl_row = Vl_row[:, None].repeat(n_agent, axis=1)  # (T + 1, a)

        masked_z = (mask * z)[:, None]
        V_row = assert_shape(jnp.maximum(jnp.max(Vhs_row, axis=-1), Vl_row - masked_z), (T + 1, n_agent))
        cat_V_row = assert_shape(jnp.concatenate([Vhs_row, Vl_row[:, :, None], V_row[:, :, None]], axis=-1),
                                 (T + 1, n_agent, nh + 2))

        Qs_GAE = assert_shape(ei.einsum(cat_V_row, gae_coeffs, "Tp1 na nhp2, Tp1 -> na nhp2"), (n_agent, nh + 2))

        # Setup Vs_row for next timestep.
        Vhs_row = Vhs_row.at[ii + 1, :].set(Vhs)
        Vl_row = Vl_row.at[ii + 1].set(Vl)

        #                            *  *        *   *             *     *
        # Update GAE coeffs. [1] -> [λ 1-λ] -> [λ² λ(1-λ) 1-λ] -> [λ³ λ²(1-λ) λ(1-λ) 1-λ]
        gae_coeffs = jnp.roll(gae_coeffs, 1)
        gae_coeffs = gae_coeffs.at[0].set(gae_lambda ** (ii + 1))
        gae_coeffs = gae_coeffs.at[1].set((gae_lambda ** ii) * (1 - gae_lambda))

        return (Vhs_row, Vl_row, gae_coeffs), Qs_GAE

    init_gae_coeffs = jnp.zeros(T + 1)
    init_gae_coeffs = init_gae_coeffs.at[0].set(1.0)

    Tah_Vh, T_Vl = Tp1ah_Vh[:-1], Tp1_Vl[:-1][:, None].repeat(n_agent, axis=1)
    Vh_final, Vl_final = Tp1ah_Vh[-1], Tp1_Vl[-1]

    init_Vhs = jnp.zeros((T + 1, n_agent, nh)).at[0, :].set(Vh_final)
    init_Vl = jnp.zeros(T + 1).at[0].set(Vl_final)[:, None].repeat(n_agent, axis=1)
    init_carry = (init_Vhs, init_Vl, init_gae_coeffs)

    ts = jnp.arange(T)[::-1]
    inps = (ts, Tah_hs, T_l, T_z, Tah_Vh, T_Vl)

    _, Qs_GAEs = jax.lax.scan(loop, init_carry, inps, reverse=True)
    Qhs_GAEs, Ql_GAEs, Q_GAEs = Qs_GAEs[:, :, :nh], Qs_GAEs[:, 0, nh], Qs_GAEs[:, :, nh + 1]
    return assert_shape(Qhs_GAEs, (T, n_agent, nh)), assert_shape(Ql_GAEs, T), assert_shape(Q_GAEs, (T, n_agent))


def compute_dec_efocp_V(z: FloatScalar, Vhs: Float[Array, "a nh"], Vl: FloatScalar) -> FloatScalar:
    assert z.shape == Vl.shape, f"z shape {z.shape} should be same as Vl shape {Vl.shape}"
    # V_i(x^τ, z; π) = max{V_i^h(o_i^τ; π), V^l(x^τ; π) - z}
    return jnp.maximum(Vhs.max(-1), (Vl - z))
    # Vhs.max(-1): max over constraints for each agent
    # Vl - z: reward value minus z
    # max: take maximum (Epigraph form)
    
def compute_conservative_exploration_gae(
    Tah_Aext: Float[Array, "T a"], # (T, n_agents) - Extrinsic advantages
    T_Delta_team_int: Float[Array, "T"], # (T,) - Team intrinsic rewards
    T_z: Float[Array, "T a"], # (T, n_agents) - Budget variables (same z per agent slot)
) -> Float[Array, "T a"]:
    """
    Compute final advantage A_t (matching InforMARL structure for extrinsic-only ablation).
    
    Note: Original EFXplorer used recursive min: A_t^{(T)} = min{A_t^ext, A_{t+1}^{(T)}, ..., A_T^ext, S_t}
    Modified for extrinsic-only ablation: A_t = A_t^ext (no min selection, matching InforMARL)
    The function structure is preserved for future intrinsic reward activation.
    
    Returns
    -------
    Tah_A_final: (T, n_agents) - Final advantages (same as Tah_Aext for extrinsic-only)
    """
    T, n_agents = Tah_Aext.shape
    
    # Step 1: Compute S_t recursively (backward) - computed but not used for extrinsic-only
    # S_t = R_t^int - z_t
    # R_t^int = sum_{k=t}^T Δ_k^team-int
    
    # Compute cumulative intrinsic returns R_t^int (backward)
    def compute_R_int(carry, inp):
        delta_team_int, z_row = inp  # delta: scalar, z_row: (n_agents,)
        R_int_next = carry
        # R_t^int = Δ_t^team-int + R_{t+1}^int
        R_int = delta_team_int + R_int_next
        # S_t = R_t^int - z_t (per agent)
        S_t = R_int - z_row  # (n_agents,)
        return R_int, S_t
    
    # Initialize: R_T^int = Δ_T^team-int
    R_int_final = T_Delta_team_int[-1]
    z_final = T_z[-1]  # (n_agents,)
    S_final = R_int_final - z_final  # (n_agents,)
    
    # Scan backward to compute all S_t (computed but not used for extrinsic-only)
    _, Tah_S = jax.lax.scan(
        compute_R_int,
        R_int_final,
        (T_Delta_team_int[:-1], T_z[:-1]),
        reverse=True
    )
    # Append final S_T
    Tah_S = jnp.concatenate([Tah_S, S_final[None, :]], axis=0)  # (T, n_agents)
    
    # Step 2: Return A_t^ext directly (matching InforMARL structure)
    # Original: A_t^{(T)} = min{A_t^ext, A_{t+1}^{(T)}, ..., A_T^ext, S_t} (recursive)
    # Modified for extrinsic-only: A_t = A_t^ext (no min selection, matching InforMARL)
    # Future: Can be changed to A_t = min{A_t^ext, S_t} when intrinsic rewards are activated
    Tah_A_final = Tah_Aext
    
    return Tah_A_final

def compute_extrinsic_gae(
    values: Float[Array, "T"],      # (T,) - Global extrinsic values
    rewards: Float[Array, "T a"],   # (T, n_agents) - Extrinsic rewards
    next_values: Float[Array, "Tp1"],  # (T+1,) - Next values
    dones: Float[Array, "T"],       # (T,) - Done flags
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Float[Array, "T a"]:
    """
    Compute extrinsic advantages A^ext(s_t, a_t)
    
    Returns
    -------
    Tah_Aext: (T, n_agents) - Extrinsic advantages
    """
    # Sum rewards over agents for global value
    T_rewards_sum = rewards.sum(axis=-1)  # (T,)
    
    # Compute global advantages
    T_targets, T_gaes = compute_gae(
        values=values,
        rewards=T_rewards_sum,
        dones=dones,
        next_values=next_values,
        gamma=gamma,
        gae_lambda=gae_lambda
    )
    
    # Broadcast to agents (same advantage for all agents in global case)
    Tah_Aext = T_gaes[:, None].repeat(rewards.shape[1], axis=1)  # (T, n_agents)
    
    return Tah_Aext