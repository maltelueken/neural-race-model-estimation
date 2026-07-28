from functools import partial
from typing import Tuple

import distrax
import jax
import jax.numpy as jnp

from .base import TruncatedNormal
from .crdm_utils import normalized_gamma_derivative
from .crdm_utils import simulate_crdm_single_trial


def create_crdm_single_prior_uniform(
    v_c_min: float = 0.0,
    v_c_max: float = 8.0,
    amp_min: float = 0.0,
    amp_max: float = 0.5,
    tau_min: float = 0.0,
    tau_max: float = 0.5,
    s_min: float = 0.0,
    s_max: float = 2.0,
    b_min: float = 0.0,
    b_max: float = 2.0,
) -> distrax.Joint:
    return distrax.Joint([
        distrax.Uniform(v_c_min, v_c_max),
        distrax.Uniform(amp_min, amp_max),
        distrax.Uniform(tau_min, tau_max),
        distrax.Uniform(s_min, s_max),
        distrax.Uniform(b_min, b_max),
    ])


def create_crdm_single_prior_informed(
    v_c_loc: float = 4.0,
    v_c_scale: float = 0.5,
    amp_loc: float = 0.3,
    amp_scale: float = 0.05,
    tau_loc: float = 0.1,
    tau_scale: float = 0.05,
    s_loc: float = 0.8,
    s_scale: float = 0.25,
    b_loc: float = 0.7,
    b_scale: float = 0.25,
) -> distrax.Joint:
    return distrax.Joint([
        TruncatedNormal(v_c_loc, v_c_scale, 0.0, jnp.inf),
        TruncatedNormal(amp_loc, amp_scale, 0.0, jnp.inf),
        TruncatedNormal(tau_loc, tau_scale, 0.0, jnp.inf),
        TruncatedNormal(s_loc, s_scale, 0.0, jnp.inf),
        TruncatedNormal(b_loc, b_scale, 0.0, jnp.inf),
    ])


@partial(jax.jit, static_argnames=["dt", "t_max"])
def simulate_crdm_single_dataset(
    keys: jnp.ndarray,
    v_c: jnp.ndarray,
    amp: jnp.ndarray,
    tau: jnp.ndarray,
    s: jnp.ndarray,
    b: jnp.ndarray,
    t0: jnp.ndarray,
    dt: float,
    t_max: float,
) -> jnp.ndarray:
    a_shape = 2.0

    t = jnp.arange(dt, t_max + dt, dt)

    v_a = normalized_gamma_derivative(t, amp, tau, a_shape)

    mu = jnp.expand_dims(v_a + v_c, axis=0)
    s = jnp.expand_dims(s, axis=(0, 1))

    rt, _ = jax.vmap(simulate_crdm_single_trial, in_axes=(None, None, None, None, None, 0))(
        mu, b, s, t0, dt, keys
    )

    return rt


@partial(jax.jit, static_argnames=["dt", "t_max"])
def simulate_crdm_single_batch(
    keys: jnp.ndarray,
    v_c: jnp.ndarray,
    amp: jnp.ndarray,
    tau: jnp.ndarray,
    s: jnp.ndarray,
    b: jnp.ndarray,
    t0: jnp.ndarray,
    dt: float,
    t_max: float,
) -> jnp.ndarray:
    in_axes = [0] * 7 + [None] * 2
    rt = jax.vmap(simulate_crdm_single_dataset, in_axes=in_axes)(
        keys, v_c, amp, tau, s, b, t0, dt, t_max,
    )

    return rt


@partial(jax.jit, static_argnames=("batch_shape", "prior", "dt", "t_max"))
def sample_conditional_crdm_single(
    key: jnp.ndarray,
    batch_shape: Tuple[int, ...],
    prior: distrax.Joint,
    dt: float,
    t_max: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    key_context, key_data = jax.random.split(key, 2)

    prior_shape = batch_shape[:-1]

    context = prior.sample(seed=key_context, sample_shape=prior_shape)

    keys = jax.random.split(key_data, batch_shape)

    x = simulate_crdm_single_batch(
        keys,
        context[0],
        context[1],
        context[2],
        context[3],
        context[4],
        jnp.zeros_like(context[0]),
        dt,
        t_max,
    )

    # No dequantisation jitter here: `simulate_crdm_single_trial` already draws
    # the crossing time uniformly within the step it was detected in, which is
    # both continuous and correctly centred.

    context = jnp.expand_dims(jnp.array(context), axis=-1)
    context = jnp.moveaxis(jnp.array(context), 0, -1)

    return jnp.expand_dims(x, -1), context
