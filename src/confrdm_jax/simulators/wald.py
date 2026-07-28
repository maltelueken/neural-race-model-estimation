"""Single accumulator with constant drift — the RDM flow's training model.

A diffusion with constant drift ``v`` and diffusion ``s`` hitting a boundary
``b`` has an inverse Gaussian first-passage time with ``mu = b / v`` and
``lam = (b / s)^2``.  So unlike every other simulator here, this one needs no
numerical integration: :func:`sample_conditional_wald` draws exactly.

That exactness has two consequences worth knowing.  There is no time grid and
therefore no ``t_max``, so this sampler can never censor and never emits the
``-1.0`` sentinel — which is why ``flows.loss_fn`` accepts ``t_max=None`` and
why ``conf_jax/model/rdm.yaml`` omits it.  And because the RDM's exact
likelihood is available analytically, the RDM path is the one where the neural
approximation can be validated against a reference posterior
(``run_reference_recovery: true``).

Parameter order: ``[v, s, b]``.  ``t0`` is absent — it is a pure location shift
applied by the likelihood, not something the flow conditions on.
"""

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
    """Wide uniform training prior over ``[v, s, b]``.

    The training prior for the RDM conditioner.  It must cover the region the
    RDM recovery prior reaches — note that the *target* accumulator's drift is
    ``v_intercept + v_slope``, so `v_max` has to bound the sum, not either
    term.  Live values are in ``conf_jax/prior/wald_uniform.yaml``.
    """
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
    """Narrow informed alternative to :func:`create_wald_prior_uniform`.

    Not referenced by any config; training on it would leave the flow
    unreliable wherever the recovery prior strays outside these bounds.
    """
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
    """Online training sampler for the RDM flow — exact inverse Gaussian draws.

    Args:
        key: PRNG key.
        batch_shape: ``(num_parameter_draws, num_trials_per_draw)``. Static.
        prior: A ``distrax.Joint`` over ``[v, s, b]``. Static (hashed by
            identity), so reuse the same object or every call recompiles.

    Returns:
        ``(data, context)`` with `data` of shape ``(B, num_trials, 1)`` holding
        decision times and `context` of shape ``(B, 1, 3)`` in natural space.
        No censoring is possible, so `data` never contains the ``-1.0``
        sentinel and ``flows.loss_fn`` may be given ``t_max=None``.
    """
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
