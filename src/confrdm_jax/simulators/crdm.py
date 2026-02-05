from functools import partial
from typing import Tuple

import distrax
import jax
import jax.numpy as jnp

from .base import TruncatedNormal
from .crdm_utils import gamma_pulse
from .crdm_utils import simulate_crdm_single_trial


@partial(jax.jit, static_argnames=["dt", "t_max"])
def simulate_crdm_dataset(
    keys: jnp.ndarray,
    v_c_intercept: jnp.ndarray,
    v_c_slope: jnp.ndarray,
    amp: jnp.ndarray,
    tau: jnp.ndarray,
    s_true: jnp.ndarray,
    b: jnp.ndarray,
    t0: jnp.ndarray,
    dt: float,
    t_max: float,
) -> jnp.ndarray:
    mu_c = jnp.hstack([v_c_intercept, v_c_intercept + v_c_slope])
    s = jnp.expand_dims(jnp.hstack([1.0, s_true]), -1)
    a_shape = 2.0

    t = jnp.arange(dt, t_max + dt, dt)

    v_a = gamma_pulse(t, jnp.abs(amp), tau, a_shape)

    mu = jnp.tile(mu_c, (t.shape[0], 1)).T

    dim_con = jnp.where(amp > 0.0, 1, 0)
    # dim_inc = jnp.where(amp > 0.0, 0, 1)

    mu = mu.at[dim_con, :].add(v_a)
    # mu = mu.at[dim_inc, :].subtract(eq4)

    rt, resp = jax.vmap(simulate_crdm_single_trial, in_axes=(None, None, None, None, None, 0))(
        mu, b, s, t0, dt, keys
    )

    return jnp.c_[rt, resp]


@partial(jax.jit, static_argnames=["dt", "t_max"])
def simulate_crdm_batch(
    keys: jnp.ndarray,
    v_c_intercept: jnp.ndarray,
    v_c_slope: jnp.ndarray,
    amp: jnp.ndarray,
    tau: jnp.ndarray,
    s_true: jnp.ndarray,
    b: jnp.ndarray,
    t0: jnp.ndarray,
    dt: float,
    t_max: float,
) -> jnp.ndarray:
    in_axes = [0] * 8 + [None] * 2
    x = jax.vmap(simulate_crdm_dataset, in_axes=in_axes)(
        keys, v_c_intercept, v_c_slope, amp, tau, s_true, b, t0, dt, t_max,
    )

    return x


def create_crdm_prior_informed(
    v_c_intercept_loc: float = 1.0,
    v_c_intercept_scale: float = 0.25,
    v_c_slope_loc: float = 4.0,
    v_c_slope_scale: float = 0.5,
    amp_loc: float = 0.3,
    amp_scale: float = 0.05,
    tau_loc: float = 0.1,
    tau_scale: float = 0.05,
    s_true_shape: float = 8.0,
    s_true_scale: float = 0.1,
    b_shape: float = 7.0,
    b_scale: float = 0.1,
    t0_loc: float = 0.3,
    t0_scale: float = 0.2,
) -> distrax.Joint:
    return distrax.Joint([
        TruncatedNormal(v_c_intercept_loc, v_c_intercept_scale, 0.0, jnp.inf),
        TruncatedNormal(v_c_slope_loc, v_c_slope_scale, 0.0, jnp.inf),
        TruncatedNormal(amp_loc, amp_scale, 0.0, jnp.inf),
        TruncatedNormal(tau_loc, tau_scale, 0.0, jnp.inf),
        distrax.Gamma(s_true_shape, 1.0 / s_true_scale),
        distrax.Gamma(b_shape, 1.0 / b_scale),
        TruncatedNormal(t0_loc, t0_scale, 0.0, jnp.inf),
    ])


@partial(jax.jit, static_argnames=("batch_shape", "prior", "dt", "t_max"))
def sample_conditional_crdm(
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

    x = simulate_crdm_batch(
        keys,
        context[0],
        context[1],
        context[2],
        context[3],
        context[4],
        context[5],
        context[6],
        dt,
        t_max,
    )

    context = jnp.expand_dims(jnp.array(context), axis=-1)
    context = jnp.moveaxis(jnp.array(context), 0, -1)

    return jnp.expand_dims(x, -1), context


@partial(jax.jit, static_argnames=("batch_shape", "prior", "dt", "t_max"))
def sample_conditional_crdm_condition(
    key: jnp.ndarray,
    batch_shape: Tuple[int, ...],
    prior: distrax.Joint,
    dt: float,
    t_max: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    key_context, key_data_con, key_data_inc = jax.random.split(key, 3)

    prior_shape = batch_shape[:-1]

    context = prior.sample(seed=key_context, sample_shape=prior_shape)

    condition = jnp.ones(batch_shape)
    condition = jnp.expand_dims(condition.at[..., (batch_shape[-1] // 2):].set(0), -1)

    data_shape = batch_shape[:-1] + (batch_shape[-1] // 2,)

    keys_con = jax.random.split(key_data_con, data_shape)

    x_con = simulate_crdm_batch(
        keys_con,
        context[0],
        context[1],
        context[2],
        context[3],
        context[4],
        context[5],
        context[6],
        dt,
        t_max,
    )

    keys_inc = jax.random.split(key_data_inc, data_shape)

    x_inc = simulate_crdm_batch(
        keys_inc,
        context[0],
        context[1],
        -context[2],
        context[3],
        context[4],
        context[5],
        context[6],
        dt,
        t_max,
    )

    x = jnp.c_[jnp.concatenate([x_con, x_inc], axis=1), condition]

    context = jnp.expand_dims(jnp.array(context), axis=-1)
    context = jnp.moveaxis(jnp.array(context), 0, -1)

    return jnp.expand_dims(x, -1), context
