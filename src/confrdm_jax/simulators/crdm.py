"""Two-accumulator conflict racing diffusion model (CRDM).

Two accumulators race to a common boundary, as in the RDM, but one of them
additionally receives a transient gamma-shaped conflict pulse.  There is no
analytical first-passage density for a time-varying drift, so these simulators
integrate the SDE by Euler-Maruyama (:func:`simulate_crdm_single_trial`).

Parameter order, positional and unenforced throughout:
``[v_c_intercept, v_c_slope, amp, tau, s_true, b, t0]``.

**Accumulator indexing.** Index 0 is the "false" / non-target accumulator with
drift ``v_c_intercept`` and diffusion fixed at 1.0; index 1 is the "true" /
target accumulator with drift ``v_c_intercept + v_c_slope`` and diffusion
``s_true``.  The diffusion of accumulator 0 is fixed because only ratios of
drift to diffusion are identified; ``s_true`` is then a relative quantity.

**Congruence coding.** The sign of `amp` selects which accumulator the pulse is
added to: ``amp > 0`` routes it to the target (congruent), ``amp < 0`` to the
non-target (incongruent).  The pulse *shape* is always
``normalized_gamma_derivative(t, |amp|, tau, 2.0)`` — the sign is a routing
switch, not a sign flip of the signal.  Downstream likelihoods rely on this:
they hand the flow ``|amp|`` and pick which accumulator it describes from the
condition column.

These are the *inference-time* simulators, used to generate test data.  The
flow itself is trained on the single-accumulator model in
``crdm_single.py``.
"""

from functools import partial
from typing import Tuple

import distrax
import jax
import jax.numpy as jnp

from .base import TruncatedNormal
from .crdm_utils import normalized_gamma_derivative
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
    """Simulate one subject's worth of CRDM trials at fixed parameters.

    Args:
        keys: One PRNG key per trial, shape ``(num_trials,)``.
        v_c_intercept: Scalar drift of the non-target accumulator.
        v_c_slope: Scalar drift advantage of the target accumulator.
        amp: Conflict pulse peak height. Its **sign selects the accumulator**
            the pulse is added to (positive = target = congruent).
        tau: Conflict pulse time scale.
        s_true: Diffusion coefficient of the target accumulator; the
            non-target's is fixed at 1.0.
        t0: Non-decision time.
        dt: Euler-Maruyama step.
        t_max: Integration horizon. Trials that have not crossed by then carry
            the sentinel ``rt = -1.0``, ``resp = -1``.

    Returns:
        Array of shape ``(num_trials, 2)`` with columns ``[rt, resp]``.
    """
    mu_c = jnp.hstack([v_c_intercept, v_c_intercept + v_c_slope])
    s = jnp.expand_dims(jnp.hstack([1.0, s_true]), -1)
    a_shape = 2.0

    t = jnp.arange(dt, t_max + dt, dt)

    v_a = normalized_gamma_derivative(t, jnp.abs(amp), tau, a_shape)

    mu = jnp.tile(mu_c, (t.shape[0], 1)).T

    # Sign of `amp` routes the pulse: positive -> target accumulator (index 1,
    # congruent), negative -> non-target (index 0, incongruent).  The pulse
    # shape itself always uses |amp|, so the two conditions differ only in
    # which accumulator is boosted, not in the signal.
    dim_con = jnp.where(amp > 0.0, 1, 0)

    mu = mu.at[dim_con, :].add(v_a)

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
    """Vectorise :func:`simulate_crdm_dataset` over a batch of parameter sets.

    Every parameter argument carries a leading batch axis of length ``B`` and
    `keys` is ``(B, num_trials)``; `dt` and `t_max` are static and shared.

    Returns:
        Array of shape ``(B, num_trials, 2)`` with columns ``[rt, resp]``.
    """
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
    """Informed CRDM prior over ``[v_c_intercept, v_c_slope, amp, tau, s_true, b, t0]``.

    This is the *recovery* prior: narrow, centred on plausible values, and used
    both to generate test data and — after adding the log-transform Jacobian —
    as the MCMC log-prior.  It must stay inside the support of the wide uniform
    prior the flow was trained over (``create_crdm_single_prior_uniform``),
    since the flow interpolates and does not extrapolate.

    The live values are declared in ``conf_jax/prior/crdm_informed.yaml``; the
    defaults here only document the intended magnitudes.  Drifts and the
    non-decision time are truncated normals on the positive half-line, `s_true`
    and `b` are gammas parameterised as ``Gamma(shape, rate = 1 / scale)``.
    """
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
    """Draw parameters from `prior` and simulate one CRDM data set per draw.

    Single-condition variant: every trial in a data set uses the same sign of
    `amp`, so all trials are congruent.  Use
    :func:`sample_conditional_crdm_condition` for the two-condition design that
    the CRDM likelihoods expect.

    Args:
        key: PRNG key.
        batch_shape: ``(num_datasets, num_trials)``. Static.
        prior: A ``distrax.Joint`` over the seven CRDM parameters. Static
            (hashed by identity), so pass the same object across calls or
            recompilation is triggered.
        dt: Euler-Maruyama step. Static.
        t_max: Integration horizon. Static.

    Returns:
        ``(data, context)`` with `data` of shape
        ``(num_datasets, num_trials, 2, 1)`` — columns ``[rt, resp]`` — and
        `context` of shape ``(num_datasets, 1, 7)`` holding the drawn
        parameters in **natural** (non-log) space.
    """
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
    """Draw parameters from `prior` and simulate a two-condition CRDM data set.

    This is the test sampler for the CRDM recovery experiments.  Each data set
    is split down the middle: the first half of the trials are congruent
    (``+amp``, pulse on the target accumulator) and the second half incongruent
    (``-amp``, pulse on the non-target).  Both halves share one parameter draw,
    so congruence is a within-subject manipulation.

    Trial order is therefore *not* randomised — all congruent trials come
    first.  Nothing downstream depends on order (the likelihood is exchangeable
    across trials and reads congruence from the third column), but any analysis
    that treats trial index as time would see a confound.

    Args:
        key: PRNG key.
        batch_shape: ``(num_datasets, num_trials)``. `num_trials` is halved for
            each condition, so an odd value silently drops a trial. Static.
        prior: A ``distrax.Joint`` over the seven CRDM parameters. Static.
        dt: Euler-Maruyama step. Static.
        t_max: Integration horizon. Static.

    Returns:
        ``(data, context)`` with `data` of shape
        ``(num_datasets, num_trials, 3, 1)`` — columns
        ``[rt, resp, condition]``, condition being 1 for congruent and 0 for
        incongruent — and `context` of shape ``(num_datasets, 1, 7)`` in
        natural space.
    """
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
