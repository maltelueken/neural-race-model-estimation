"""Two-accumulator racing diffusion model (RDM).

Two independent accumulators with constant drift race to a common boundary;
the first to arrive determines both the response and the response time.  Each
accumulator's first-passage time is inverse Gaussian, so the race can be
simulated exactly by drawing one per accumulator and taking the minimum — no
time discretisation anywhere.

Parameter order: ``[v_intercept, v_slope, s_true, b, t0]``.

**Accumulator indexing**, shared with the CRDM: index 0 is the "false" /
non-target accumulator with drift ``v_intercept`` and diffusion fixed at 1.0;
index 1 is the "true" / target accumulator with drift
``v_intercept + v_slope`` and diffusion ``s_true``.  Fixing one diffusion
coefficient is what makes the rest identifiable — only drift-to-diffusion
ratios are — so ``s_true`` is a relative quantity, not an absolute noise level.

Because the exact likelihood is available (``likelihoods.rdm``), the RDM is the
model used to validate the neural approximation against a reference posterior.
"""

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
    """Informed RDM prior over ``[v_intercept, v_slope, s_true, b, t0]``.

    The *recovery* prior: it generates the test data and, with the
    log-transform Jacobian added, serves as the MCMC log-prior — one object,
    so the two can never drift apart.  Live values are in
    ``conf_jax/prior/rdm_informed.yaml``.

    Its support has to sit inside the flow's training box
    (``create_wald_prior_uniform``), and the binding constraint is on the
    *target* accumulator: its drift is ``v_intercept + v_slope``, i.e. ~2.5
    under these defaults against a training range of ``[0, 8]``.

    `s_true` and `b` are gammas parameterised as
    ``Gamma(shape, rate = 1 / scale)``, so ``s_true_shape=12``,
    ``s_true_scale=0.1`` means a mean of 1.2, not 12.
    """
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
    """Simulate an RDM race by drawing each accumulator's first-passage time.

    Exact, not discretised: a constant-drift diffusion hits `b` at an inverse
    Gaussian time with ``mu = b / v`` and ``lam = (b / s)^2``, so the race is
    the minimum of two such draws.  Consequently no trial can be censored and
    the output never contains the ``-1.0`` sentinel.

    Every parameter may be a scalar or carry a trailing batch axis, which is
    broadcast through; the accumulator axis is prepended.

    Args:
        v_intercept: Drift of the non-target accumulator (index 0).
        v_slope: Drift advantage of the target accumulator (index 1).
        s_true: Diffusion of the target accumulator; the non-target's is 1.0.
        b: Absorbing boundary.
        t0: Non-decision time, added to the winning first-passage time.
        batch_shape: Number of trials to draw.
        key: PRNG key.

    Returns:
        Array with columns ``[rt, resp]`` stacked on the last axis, `resp`
        being the index of the winning accumulator. Shape is
        ``(num_trials, 2)`` for scalar parameters and
        ``(batch, num_trials, 2)`` when they carry a batch axis.
    """
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
    """Draw parameters from `prior` and simulate one RDM data set per draw.

    The test sampler for the single-subject RDM recovery experiments.

    Args:
        key: PRNG key.
        batch_shape: ``(num_datasets, num_trials)``. Static.
        prior: A ``distrax.Joint`` over the five RDM parameters. Static.

    Returns:
        ``(data, context)`` with `data` of shape
        ``(num_datasets, num_trials, 2)`` — columns ``[rt, resp]``, no trailing
        singleton unlike the CRDM samplers — and `context` of shape
        ``(num_datasets, 1, 5)`` in natural space.
    """
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
