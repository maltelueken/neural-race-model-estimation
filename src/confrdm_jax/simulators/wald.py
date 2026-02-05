from functools import partial
from typing import Tuple

import distrax
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions

from .base import TruncatedNormal


def create_wald_prior_uniform(
    v_min: float = 0.0,
    v_max: float = 8.0,
    s_min: float = 0.0,
    s_max: float = 2.0,
    b_min: float = 0.0,
    b_max: float = 2.0,
) -> distrax.Joint:
    return distrax.Joint([
        distrax.Uniform(v_min, v_max),
        distrax.Uniform(s_min, s_max),
        distrax.Uniform(b_min, b_max),
    ])


def create_wald_prior_informed(
    v_loc: float = 4.0,
    v_scale: float = 0.5,
    s_loc: float = 1.0,
    s_scale: float = 0.5,
    b_loc: float = 1.0,
    b_scale: float = 0.5,
) -> distrax.Joint:
    return distrax.Joint([
        TruncatedNormal(v_loc, v_scale, 0.0, jnp.inf),
        TruncatedNormal(s_loc, s_scale, 0.0, jnp.inf),
        TruncatedNormal(b_loc, b_scale, 0.0, jnp.inf),
    ])


@partial(jax.jit, static_argnames=("batch_shape", "prior"))
def sample_conditional_wald(
    key: jnp.ndarray,
    batch_shape: Tuple[int, ...],
    prior: distrax.Joint,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    key_context, key_data = jax.random.split(key, 2)

    prior_shape = batch_shape[:-1]

    context = prior.sample(seed=key_context, sample_shape=prior_shape)

    # mu = b/drift
    mu = context[2] / context[0]
    # lam = (b/s)**2
    lam = (context[2] / context[1]) ** 2

    data = distributions.InverseGaussian(mu, lam).sample(batch_shape[-1], key_data)

    context = jnp.expand_dims(jnp.array(context), axis=-1)
    context = jnp.moveaxis(jnp.array(context), 0, -1)

    return jnp.expand_dims(jnp.moveaxis(data, 0, -1), -1), context
