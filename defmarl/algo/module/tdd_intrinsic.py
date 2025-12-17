import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from typing import Callable, Optional, Tuple
from flax.training.train_state import TrainState


# -------------------------
# Distance and MLP blocks
# -------------------------

def mrn_distance(x: jnp.ndarray, y: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """
    Metric Residual Network (MRN) distance.
    x, y: (..., d)
    returns: (...) distance
    """
    d = x.shape[-1]
    x_prefix, x_suffix = x[..., : d // 2], x[..., d // 2 :]
    y_prefix, y_suffix = y[..., : d // 2], y[..., d // 2 :]
    max_component = jnp.maximum(0.0, x_prefix - y_prefix).max(axis=-1)
    l2_component = jnp.sqrt(jnp.square(x_suffix - y_suffix).sum(axis=-1) + eps)
    return max_component + l2_component


class MLP(nn.Module):
    hidden_dim: int
    out_dim: int
    layers: int = 2
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        for _ in range(self.layers - 1):
            x = nn.Dense(self.hidden_dim)(x)
            x = self.activation(x)
        x = nn.Dense(self.out_dim)(x)
        return x


class Encoder(nn.Module):
    hidden_dim: int = 256
    out_dim: int = 256
    layers: int = 2
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, obs):
        return MLP(self.hidden_dim, self.out_dim, self.layers, self.activation)(obs)


class PotentialNet(nn.Module):
    hidden_dim: int = 256
    layers: int = 2
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, obs):
        return MLP(self.hidden_dim, 1, self.layers, self.activation)(obs)


# -------------------------
# TDD forward / loss / reward
# -------------------------

def tdd_forward(params, apply_enc, apply_pot, obs_t, obs_tp1, energy_fn: str = "mrn_pot", temperature: float = 1.0):
    """
    obs_t, obs_tp1: (B, obs_dim)
    returns logits (B, B), distances (B, B)
    """
    phi_x = apply_enc(params["enc"], obs_t)   # (B, d)
    phi_y = apply_enc(params["enc"], obs_tp1) # (B, d)
    c_y = apply_pot(params["pot"], obs_tp1)   # (B, 1)

    if energy_fn == "l2":
        logits = -jnp.sqrt(((phi_x[:, None] - phi_y[None, :]) ** 2).sum(axis=-1) + 1e-8)
    elif energy_fn == "cos":
        phi_x_norm = phi_x / (jnp.linalg.norm(phi_x, axis=-1, keepdims=True) + 1e-8)
        phi_y_norm = phi_y / (jnp.linalg.norm(phi_y, axis=-1, keepdims=True) + 1e-8)
        phi_x_norm = phi_x_norm / temperature
        logits = jnp.einsum("ik,jk->ij", phi_x_norm, phi_y_norm)
    elif energy_fn == "dot":
        logits = jnp.einsum("ik,jk->ij", phi_x, phi_y)
    elif energy_fn == "mrn":
        logits = -mrn_distance(phi_x[:, None], phi_y[None, :])
    else:  # "mrn_pot" default
        logits = c_y.T - mrn_distance(phi_x[:, None], phi_y[None, :])

    return logits


def tdd_loss(params, apply_enc, apply_pot, obs_t, obs_tp1, energy_fn="mrn_pot", loss_fn="infonce", temperature=1.0):
    logits = tdd_forward(params, apply_enc, apply_pot, obs_t, obs_tp1, energy_fn=energy_fn, temperature=temperature)
    b = logits.shape[0]
    eye = jnp.eye(b)

    def ce(log):
        return jnp.mean(optax.softmax_cross_entropy(log, eye))

    if loss_fn == "infonce":
        loss = ce(logits)
    elif loss_fn == "infonce_backward":
        loss = ce(logits.T)
    elif loss_fn == "infonce_symmetric":
        loss = 0.5 * (ce(logits) + ce(logits.T))
    else:
        loss = ce(logits)  # default

    metrics = {
        "tdd/loss": loss,
        "tdd/logits_pos": jnp.mean(jnp.diag(logits)),
        "tdd/logits_neg": jnp.mean(logits * (1 - eye)),
        "tdd/logits_logsumexp": jnp.mean(jnp.logsumexp(logits + 1e-6, axis=1) ** 2),
        "tdd/acc": jnp.mean(jnp.argmax(logits, axis=1) == jnp.arange(b)),
    }
    return loss, metrics


def tdd_intrinsic_reward(params, apply_enc, obs_t, obs_tp1, aggregate="min"):
    """
    obs_t, obs_tp1: (B, obs_dim)
    returns: intrinsic_rewards (B,)
    """
    phi_x = apply_enc(params["enc"], obs_t)
    phi_y = apply_enc(params["enc"], obs_tp1)
    dists = mrn_distance(phi_x, phi_y)  # (B,)

    if aggregate == "min":
        rew = dists
    elif aggregate == "quantile10":
        q = jnp.quantile(dists, 0.1)
        rew = jnp.full_like(dists, q)
    else:  # knn10 or default
        k = 10
        k = jnp.minimum(k, dists.shape[0])
        knn = jnp.sort(dists)[k - 1]
        rew = jnp.full_like(dists, knn)

    return rew


# -------------------------
# TrainState helpers
# -------------------------

class TDDTrainState(TrainState):
    apply_fn_enc: Callable = None
    apply_fn_pot: Callable = None


def create_tdd_train_state(rng, obs_dim: int, hidden_dim: int = 256, lr: float = 3e-4,
                           energy_fn: str = "mrn_pot", loss_fn: str = "infonce", temperature: float = 1.0):
    enc = Encoder(hidden_dim=hidden_dim, out_dim=hidden_dim)
    pot = PotentialNet(hidden_dim=hidden_dim)

    rng_enc, rng_pot = jax.random.split(rng)
    dummy = jnp.zeros((1, obs_dim))
    enc_params = enc.init(rng_enc, dummy)
    pot_params = pot.init(rng_pot, dummy)

    params = {"enc": enc_params, "pot": pot_params}
    tx = optax.adam(lr)

    return TDDTrainState.create(
        apply_fn=None,
        params=params,
        tx=tx,
        apply_fn_enc=enc.apply,
        apply_fn_pot=pot.apply,
    )


@jax.jit
def tdd_train_step(state: TDDTrainState, obs_t: jnp.ndarray, obs_tp1: jnp.ndarray,
                   energy_fn: str = "mrn_pot", loss_fn: str = "infonce", temperature: float = 1.0):
    def loss_fn_(params):
        loss, metrics = tdd_loss(
            params,
            state.apply_fn_enc,
            state.apply_fn_pot,
            obs_t,
            obs_tp1,
            energy_fn=energy_fn,
            loss_fn=loss_fn,
            temperature=temperature
        )
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn_, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, metrics





