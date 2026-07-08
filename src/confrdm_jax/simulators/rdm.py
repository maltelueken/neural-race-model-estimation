from functools import partial
from typing import Tuple

import distrax
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd


def create_rdm_prior_informed(
    v_intercept_loc: float = 1.0,
    v_intercept_scale: float = 0.25,
    v_scale_loc: float = 1.5,
    v_scale_scale: float = 0.5,
    s_true_shape: float = 12.0,
    s_true_scale: float = 0.1,
    b_shape: float = 8.0,
    b_scale: float = 0.15,
    t0_loc: float = 0.3,
    t0_scale: float = 0.2,
) -> distrax.Joint:
    return distrax.Joint([
        tfd.TruncatedNormal(v_intercept_loc, v_intercept_scale, 0.0, jnp.inf),
        tfd.TruncatedNormal(v_scale_loc, v_scale_scale, 0.0, jnp.inf),
        distrax.Gamma(s_true_shape, 1.0 / s_true_scale),
        distrax.Gamma(b_shape, 1.0 / b_scale),
        tfd.TruncatedNormal(t0_loc, t0_scale, 0.0, jnp.inf),
    ])


def simulate_rdm(
    v_intercept: jnp.ndarray,
    v_slope: jnp.ndarray,
    s_true: jnp.ndarray,
    b: jnp.ndarray,
    t0: jnp.ndarray,
    batch_shape: Tuple[int, ...],
    key: jnp.ndarray,
) -> jnp.ndarray:

    v = jnp.stack([v_intercept, v_intercept + v_slope], axis=0)
    s = jnp.stack([jnp.ones_like(s_true), s_true], axis=0)

    mu = b / v
    lam = (b / s) ** 2

    fpt = tfd.InverseGaussian(mu, lam).sample(batch_shape, key)

    resp = jnp.argmin(fpt, axis=1)
    rt = jnp.min(fpt, axis=1) + t0
    rt = jnp.transpose(rt)
    resp = jnp.transpose(resp)

    return jnp.stack([rt, resp], axis=-1)


@partial(jax.jit, static_argnames=("batch_shape", "prior"))
def sample_conditional_rdm(
    key: jnp.ndarray,
    batch_shape: Tuple[int, ...],
    prior: distrax.Joint,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    key_context, key_data = jax.random.split(key, 2)

    prior_shape = batch_shape[:-1]

    context = prior.sample(seed=key_context, sample_shape=prior_shape)

    data = simulate_rdm(
        context[0],
        context[1],
        context[2],
        context[3],
        context[4],
        batch_shape[-1],
        key_data,
    )

    context = jnp.expand_dims(jnp.array(context), axis=-1)
    context = jnp.moveaxis(jnp.array(context), 0, -1)

    return data, context
