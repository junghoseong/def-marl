import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Dict

# $\gamma$-discounted sampling, K ~ Geom(1-discount), gamma^k(1-gamma)
def discounted_sampling(ranges, discount, rng):
    """
    Discounted sampling for state selection.
    
    Args:
        ranges: (jnp.ndarray) shape (batch_size,) - maximum steps to sample
        discount: (float) discount factor [0, 1]
        rng: (jax.random.PRNGKey) random key
        
    Returns:
        samples: (jnp.ndarray) shape (batch_size,) - sampled step indices
    """

    seeds = jax.random.uniform(rng, shape=ranges.shape)
    
    if discount == 0:
        samples = jnp.zeros_like(seeds, dtype=jnp.int32)
    elif discount == 1:
        samples = jnp.floor(seeds * ranges).astype(jnp.int32)
    else:
        samples = jnp.log(1 - (1 - discount**ranges) * seeds) / jnp.log(discount)
        samples = jnp.minimum(jnp.floor(samples).astype(jnp.int32), ranges - 1)
    
    return samples

# Metric Residual Network (MRN) distance
def mrn_distance(x, y):
    """
    Metric Residual Network (MRN) architecture. (https://arxiv.org/abs/2208.08133)
    
    Args:
        x: (jnp.ndarray) tensor of shape (..., dim) phi_s = state_encoder(s)
        y: (jnp.ndarray) tensor of shape (..., dim) phi_sp = state_encoder(s_p)
        
    Returns:
        distance: (jnp.ndarray) tensor of shape - MRN distance
    """
    eps = 1e-6
    
    d = x.shape[-1]
    x_prefix = x[..., :d // 2]
    x_suffix = x[..., d // 2:]
    y_prefix = y[..., :d // 2]
    y_suffix = y[..., d // 2:]
    
    # Vectorized operations
    max_component = jnp.maximum(0, x_prefix - y_prefix).max(axis=-1)
    l2_component = jnp.sqrt(jnp.square(x_suffix - y_suffix).sum(axis=-1) + eps)
    
    distance = max_component + l2_component
    
    return distance

class PotentialNet(nn.Module):
    """
    Potential network for training MRN.
    
    Input: agent_local_features (2-dim: position 2)
        cf. global-mode (4-dim: velocity 2 + position 2)
    Output: potemtial value (1-dim)
    """
    latent_dim: int
    
    @nn.compact
    def __call__(self, obs):
        """
        Args:
            obs: (jnp.ndarray) shape (..., input_dim) - agent local features
            
        Returns:
            value: (jnp.ndarray) shape (..., 1) - potential value
        """
        x = nn.Dense(self.latent_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(obs)
        x = nn.relu(x)
        x = nn.Dense(self.latent_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return value

class state_encoder(nn.Module):
    """
    state encoder for training MRN.
    
    Input: agent_local_features (2-dim: position 2)
        cf. global-mode (4-dim: velocity 2 + position 2)
    Output: latent representation (output_dim)
    """
    latent_dim: int
    output_dim: int
    
    @nn.compact
    def __call__(self, obs):
        """
        Args:
            obs: (jnp.ndarray) shape (..., input_dim) - agent local features
            
        Returns:
            encoded: (jnp.ndarray) shape (..., output_dim) - latent representation
        """
        x = nn.Dense(self.latent_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(obs)
        x = nn.relu(x)
        x = nn.Dense(self.latent_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        encoded = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return encoded

class IntrinsicRewardModel(nn.Module):
    """
    Training MRN+Potential network for intrinsic reward
    through contrastive learning.
    """

    def __init__(self):
        self.potential_net = PotentialNet(latent_dim=256)
        self.state_encoder = state_encoder(latent_dim=256, output_dim=256)

    def forward(self, curr_obs, future_obs):
        phi_x = self.state_encoder(curr_obs)
        phi_y = self.state_encoder(future_obs)
        c_y = self.potential_net(future_obs)

        logits = c_y.T - mrn_distance(phi_x[:, None], phi_y[None, :])

        batch_size = logits.size(0)
        I = jnp.eye(batch_size)
        contrastive_loss = jax.nn.softmax_cross_entropy(logits, I)
        contrastive_loss = jnp.mean(contrastive_loss)

        logs = {
            'contrastive_loss': contrastive_loss,
            'logits_pos': jnp.diag(logits).mean(),
            'logits_neg': jnp.mean(logits * (1 - I)),
            'logits_logsumexp': jnp.mean((jnp.logsumexp(logits + 1e-6, axis=1)**2)),
            'categorical_accuracy': jnp.mean((jnp.argmax(logits, axis=1) == jnp.arange(batch_size)).float()),
        }

        return contrastive_loss, logs

    def get_intrinsic_rewards(self,
        curr_obs, next_obs, last_mems, curr_act, curr_dones, obs_history, stats_logger  
    ):
        with jax.no_grad():
            batch_size = curr_obs.size(0)
            int_rews = jnp.zeros(batch_size)

            for env_id in range(batch_size):
                curr_obs_emb = curr_obs[env_id].reshape(1, -1)
                next_obs_emb = next_obs[env_id].reshape(1, -1)
                obs_embs = obs_history[env_id]
                new_embs = [curr_obs_emb, next_obs_emb] if obs_embs is None else [obs_embs, next_obs_emb]
                obs_embs = jnp.concatenate(new_embs, axis=0)
                obs_history[env_id] = obs_embs
                phi_x = self.state_encoder(obs_embs[:-1])
                phi_y = self.state_encoder(obs_embs[-1].reshape(1, -1))
                dists = mrn_distance(phi_x, phi_y)
                int_rews[env_id] += dists.min().item

        logs = {
            'int_rews_mean': int_rews.mean(),
            'int_rews_min': int_rews.min(),
            'int_rews_max': int_rews.max(),
            'int_rews_std': int_rews.std(),
        }
        return int_rews, logs