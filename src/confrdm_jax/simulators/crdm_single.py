"""Single-accumulator conflict diffusion — the CRDM flow's training model.

The neural likelihood estimator is amortised over parameters, not over data
sets: it learns the first-passage density of **one** accumulator subject to a
conflict pulse, and the two-accumulator race in ``crdm.py`` is reassembled
analytically at inference time.  This module is that one accumulator.

Differences from the two-accumulator simulators, all deliberate:

- ``t0`` is fixed at 0 by :func:`sample_conditional_crdm_single`, so the flow
  models decision times and ``t0`` becomes exactly a location shift applied by
  the likelihood at ``rt - t0``.  It is not a conditioning variable.
- ``amp`` is used as-is rather than sign-routed; the likelihoods pass
  ``jnp.abs(amp)`` and decide separately which accumulator the pulse belongs to.
- The prior is wide and uniform (``create_crdm_single_prior_uniform``).  The
  flow interpolates, so it has to be trained over a superset of the region MCMC
  will visit under the narrow ``crdm_informed`` recovery prior.

Parameter order: ``[v_c, amp, tau, s, b]`` — five conditioning variables, which
is why ``conf_jax/model/crdm.yaml`` sets ``num_params: 5`` for the conditioner.
"""

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
    """Wide uniform training prior over ``[v_c, amp, tau, s, b]``.

    Deliberately much wider than the recovery prior: the flow is an
    interpolator, so its training support must cover every parameter
    combination MCMC can visit.  Live values are in
    ``conf_jax/prior/crdm_single_uniform.yaml``, which widens `amp`, `tau`, `s`
    and `b` beyond the defaults here.

    The low-drift / high-boundary corner of this box is where trials fail to
    cross within ``t_max`` — about 3% of trials overall, heavily concentrated
    there.  Those are handled as right-censored observations by
    ``flows.loss_fn`` rather than dropped.
    """
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
    """Narrow informed alternative to :func:`create_crdm_single_prior_uniform`.

    Not referenced by any config — kept for focused experiments where training
    the flow only over the plausible region is enough.  Training on it would
    make the flow unusable for MCMC under a wider recovery prior.
    """
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
    """Simulate first-passage times for one accumulator at fixed parameters.

    Args:
        keys: One PRNG key per trial, shape ``(num_trials,)``.
        v_c: Constant drift component.
        amp: Conflict pulse peak height, used as given (no sign routing).
        tau: Conflict pulse time scale.
        s: Diffusion coefficient.
        b: Absorbing boundary.
        t0: Non-decision time; callers training the flow pass 0.
        dt: Euler-Maruyama step. Static.
        t_max: Integration horizon. Static.

    Returns:
        First-passage times of shape ``(num_trials,)``, with ``-1.0`` marking
        trials that never crossed.
    """
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
    """Vectorise :func:`simulate_crdm_single_dataset` over a batch of parameters.

    Every parameter argument carries a leading batch axis of length ``B`` and
    `keys` is ``(B, num_trials)``.

    Returns:
        First-passage times of shape ``(B, num_trials)``.
    """
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
    """Online training sampler for the CRDM flow.

    Called once per training step; nothing is stored between steps.  ``t0`` is
    forced to 0 so the flow learns decision times only.

    Args:
        key: PRNG key.
        batch_shape: ``(num_parameter_draws, num_trials_per_draw)``. Static.
        prior: A ``distrax.Joint`` over ``[v_c, amp, tau, s, b]``. Static
            (hashed by identity).
        dt: Euler-Maruyama step. Static.
        t_max: Integration horizon. Static; ``flows.loss_fn`` must be given the
            same value so censored trials are scored by ``log S(t_max)``.

    Returns:
        ``(data, context)`` with `data` of shape ``(B, num_trials, 1)`` holding
        decision times (``-1.0`` where censored) and `context` of shape
        ``(B, 1, 5)`` in natural space.
    """
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
