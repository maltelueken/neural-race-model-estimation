"""Simulators: prior draws plus synthetic data, for training and for recovery.

Two jobs, with different shapes:

* **Training samplers** (:func:`sample_conditional_wald`,
  :func:`sample_conditional_crdm_single`) feed the online loop in
  ``scripts/train_conditioner.py``. They draw one accumulator's first-passage times, because
  that is what the flow learns; the race is reassembled analytically at inference time. They
  force ``t0 = 0``, so the flow models decision times and ``t0`` becomes exactly a location
  shift applied by the race at ``rt - t0`` — one fewer conditioning dimension.
* **Test samplers** (:func:`sample_conditional_rdm`,
  :func:`sample_conditional_crdm_condition`, and the two hierarchical ones) generate the
  data the recovery scripts fit. They run the full race through
  :func:`eamax.simulate.simulate_race`, driven by the *same*
  :class:`eamax.design.Parameterization` the likelihood scores with, so the generative and
  inferential models cannot drift apart.

**Censoring.** A racing accumulator that never crosses within ``t_max`` returns ``inf``, and
:func:`eamax.simulate.race_sample` turns an all-``inf`` trial into the ``(-1.0, -1)``
sentinel that :func:`eamax.race.race_loglik` reads back as a right-censored observation. The
single-accumulator training samplers have no race, so they pass ``inf`` through unchanged —
``eamax.flows.loss_fn`` treats a non-finite or non-positive decision time as censored and
scores it by ``log S(t_max)``.

**Congruency is a design column, not a sign.** The old CRDM simulators routed the conflict
pulse by the sign of ``amp``. It is now the design's ``distractor`` column, which under the
two-condition design is the congruency indicator itself; ``amp`` is non-negative and simply
zero on accumulators the pulse does not ride. See :mod:`confrdm_jax.specs`.

**Numerical note.** The pulsed accumulator's grid values now come from the pulse's
*closed-form* integrated drift (:class:`eamax.accumulators.SimulatedPulsedWald`) rather than
from an Euler–Maruyama Riemann sum, which removes the sum's ``O(dt)`` drift error at the
same ``dt``. The Brownian-bridge crossing test and within-step dequantisation are unchanged.
CRDM draws therefore differ from pre-migration ones, by more than reseeding, and are more
accurate at any given ``dt``.
"""

from functools import partial
from typing import Tuple

import distrax
import jax
import jax.numpy as jnp
from eamax.accumulators import SimulatedPulsedWald, Wald
from eamax.design import TrialDesign, build_params_fn
from eamax.hierarchical import reconstruct_from_dict
from eamax.simulate import simulate_race

from .priors import (
    create_crdm_prior_informed,
    create_crdm_single_prior_informed,
    create_crdm_single_prior_uniform,
    create_hierarchical_crdm_prior_lkj_mvn,
    create_hierarchical_rdm_prior_lkj_mvn,
    create_rdm_prior_informed,
    create_wald_prior_informed,
    create_wald_prior_uniform,
)
from .specs import crdm_spec, rdm_spec

__all__ = [
    # Priors are re-exported so `conf_jax/prior/*.yaml` and `conf_jax/model/*.yaml` keep
    # one `_target_` namespace for everything generative.
    "create_crdm_prior_informed",
    "create_crdm_single_prior_informed",
    "create_crdm_single_prior_uniform",
    "create_hierarchical_crdm_prior_lkj_mvn",
    "create_hierarchical_rdm_prior_lkj_mvn",
    "create_rdm_prior_informed",
    "create_wald_prior_informed",
    "create_wald_prior_uniform",
    # Designs and single-dataset simulators
    "crdm_condition_design",
    "rdm_design",
    "simulate_crdm",
    "simulate_rdm",
    # Samplers
    "sample_conditional_crdm_condition",
    "sample_conditional_crdm_hierarchical_lkj_mvn",
    "sample_conditional_crdm_single",
    "sample_conditional_rdm",
    "sample_conditional_rdm_hierarchical_lkj_mvn",
    "sample_conditional_wald",
]


def _stack_context(context):
    """A ``distrax.Joint`` draw of shape ``(P,) x (B,)`` to the ``(B, 1, P)`` the flow wants.

    The middle axis is a broadcasting axis, not data: the conditioner maps it to one set of
    spline knots per parameter draw, which then broadcasts across that draw's trials.
    """
    return jnp.moveaxis(jnp.expand_dims(jnp.asarray(context), axis=-1), 0, -1)


# --------------------------------------------------------------------------------------- #
# Trial designs
# --------------------------------------------------------------------------------------- #


def rdm_design(num_trials):
    """A :class:`eamax.design.TrialDesign` for `num_trials` RDM trials.

    The only covariate is ``target``, which is all ones: the target accumulator is index 1 on
    every trial. ``rt`` and ``response`` are placeholders — they are what is being generated.
    """
    zeros = jnp.zeros(num_trials)
    return TrialDesign(rt=zeros, target=jnp.ones(num_trials), first_response=0)


def crdm_condition_design(num_trials):
    """A two-condition CRDM design: congruent first half, incongruent second.

    ``condition`` is 1 for congruent and 0 for incongruent, and doubles as ``distractor`` —
    the accumulator the conflict pulse rides. A congruent trial's distracting feature points
    at the target (accumulator 1), an incongruent one's at the non-target (accumulator 0).

    Trial order is *not* randomised: all congruent trials come first. Nothing downstream
    depends on order (the likelihood is exchangeable across trials and reads congruency from
    the third data column), but any analysis treating trial index as time would see a
    confound. An odd `num_trials` gives the extra trial to the congruent half.
    """
    condition = (jnp.arange(num_trials) < (num_trials - num_trials // 2)).astype(float)
    return TrialDesign(
        rt=jnp.zeros(num_trials),
        target=jnp.ones(num_trials),
        condition=condition,
        distractor=condition,
        first_response=0,
    )


# --------------------------------------------------------------------------------------- #
# Single-dataset simulators
# --------------------------------------------------------------------------------------- #


def simulate_rdm(key, log_theta, num_trials):
    """One subject's RDM trials, drawn exactly.

    Each accumulator's first-passage time is inverse Gaussian, so the race is the minimum of
    two exact draws — no time discretisation, and therefore no trial can be censored.

    Args:
        key: PRNG key.
        log_theta: ``(5,)`` log-parameters in :data:`confrdm_jax.specs.RDM_PARAM_NAMES` order.
        num_trials: Trials to draw.

    Returns:
        ``(num_trials, 2)`` with columns ``[rt, choice]``.
    """
    spec = rdm_spec()
    design = rdm_design(num_trials)
    rt, response = simulate_race(key, build_params_fn(spec, Wald()), log_theta, design, Wald())
    return jnp.stack([rt, response], axis=-1)


def simulate_crdm(key, log_theta, num_trials, dt, t_max, chunk_size=None):
    """One subject's two-condition CRDM trials, simulated on a grid.

    Args:
        key: PRNG key.
        log_theta: ``(7,)`` log-parameters in :data:`confrdm_jax.specs.CRDM_PARAM_NAMES`
            order.
        num_trials: Trials to draw; the first half congruent, the rest incongruent.
        dt: Grid step.
        t_max: Integration horizon. Trials with no crossing by then carry ``rt = -1.0``,
            ``choice = -1``.
        chunk_size: Maximum elements per integration call, to bound peak memory.

    Returns:
        ``(num_trials, 3)`` with columns ``[rt, choice, condition]``.
    """
    spec = crdm_spec()
    design = crdm_condition_design(num_trials)
    accumulator = SimulatedPulsedWald(dt=dt, t_max=t_max, chunk_size=chunk_size)
    rt, response = simulate_race(
        key, build_params_fn(spec, accumulator), log_theta, design, accumulator
    )
    return jnp.stack([rt, response, design.condition], axis=-1)


# --------------------------------------------------------------------------------------- #
# Training samplers: one accumulator, t0 = 0, called once per training step.
# --------------------------------------------------------------------------------------- #


@partial(jax.jit, static_argnames=("batch_shape", "prior"))
def sample_conditional_wald(
    key: jnp.ndarray,
    batch_shape: Tuple[int, ...],
    prior: distrax.Joint,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Online training sampler for the RDM flow — exact inverse Gaussian draws.

    There is no time grid and therefore no ``t_max``, so this sampler can never censor, which
    is why ``eamax.flows.loss_fn`` may be given ``t_max=None`` and ``conf_jax/model/rdm.yaml``
    omits it.

    Args:
        key: PRNG key.
        batch_shape: ``(num_parameter_draws, num_trials_per_draw)``. Static.
        prior: A ``distrax.Joint`` over ``[v, s, b]``. Static (hashed by identity), so reuse
            the same object or every call recompiles.

    Returns:
        ``(data, context)`` with `data` of shape ``(B, num_trials, 1)`` holding decision
        times and `context` of shape ``(B, 1, 3)`` in natural space.
    """
    key_context, key_data = jax.random.split(key, 2)

    context = prior.sample(seed=key_context, sample_shape=batch_shape[:-1])
    v, s, b = context

    params = {
        name: jnp.broadcast_to(value[:, None], batch_shape)
        for name, value in (("v", v), ("s", s), ("b", b))
    }
    data = Wald().sample(key_data, params)

    return jnp.expand_dims(data, -1), _stack_context(context)


@partial(jax.jit, static_argnames=("batch_shape", "prior", "dt", "t_max", "chunk_size"))
def sample_conditional_crdm_single(
    key: jnp.ndarray,
    batch_shape: Tuple[int, ...],
    prior: distrax.Joint,
    dt: float,
    t_max: float,
    chunk_size: int | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Online training sampler for the CRDM flow — one pulsed accumulator on a grid.

    Called once per training step; nothing is stored between steps. ``amp`` is used as drawn
    rather than sign-routed: the flow learns the density of a pulsed accumulator, and which
    accumulator carries the pulse is the race's business.

    Args:
        key: PRNG key.
        batch_shape: ``(num_parameter_draws, num_trials_per_draw)``. Static.
        prior: A ``distrax.Joint`` over ``[v_c, amp, tau, s, b]``. Static (hashed by
            identity).
        dt: Grid step. Static.
        t_max: Integration horizon. Static; ``eamax.flows.loss_fn`` must be given the same
            value so censored trials are scored by ``log S(t_max)``.
        chunk_size: Maximum elements per integration call, to bound peak memory. Static.

    Returns:
        ``(data, context)`` with `data` of shape ``(B, num_trials, 1)`` holding decision
        times — ``inf`` where the accumulator never crossed — and `context` of shape
        ``(B, 1, 5)`` in natural space.
    """
    key_context, key_data = jax.random.split(key, 2)

    context = prior.sample(seed=key_context, sample_shape=batch_shape[:-1])
    names = ("v", "amp", "tau", "s", "b")

    params = {
        name: jnp.broadcast_to(value[:, None], batch_shape)
        for name, value in zip(names, context)
    }
    accumulator = SimulatedPulsedWald(dt=dt, t_max=t_max, chunk_size=chunk_size)
    data = accumulator.sample(key_data, params)

    return jnp.expand_dims(data, -1), _stack_context(context)


# --------------------------------------------------------------------------------------- #
# Test samplers: the full race, for the recovery scripts.
# --------------------------------------------------------------------------------------- #


@partial(jax.jit, static_argnames=("batch_shape", "prior"))
def sample_conditional_rdm(
    key: jnp.ndarray,
    batch_shape: Tuple[int, ...],
    prior: distrax.Joint,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Draw parameters from `prior` and simulate one RDM dataset per draw.

    Args:
        key: PRNG key.
        batch_shape: ``(num_datasets, num_trials)``. Static.
        prior: A ``distrax.Joint`` over the five RDM parameters. Static.

    Returns:
        ``(data, context)`` with `data` of shape ``(num_datasets, num_trials, 2)`` — columns
        ``[rt, choice]`` — and `context` of shape ``(num_datasets, 5)`` in natural space.
    """
    key_context, key_data = jax.random.split(key, 2)

    num_datasets, num_trials = batch_shape
    context = jnp.stack(prior.sample(seed=key_context, sample_shape=(num_datasets,)), axis=-1)

    data = jax.vmap(simulate_rdm, in_axes=(0, 0, None))(
        jax.random.split(key_data, num_datasets), jnp.log(context), num_trials
    )
    return data, context


@partial(jax.jit, static_argnames=("batch_shape", "prior", "dt", "t_max", "chunk_size"))
def sample_conditional_crdm_condition(
    key: jnp.ndarray,
    batch_shape: Tuple[int, ...],
    prior: distrax.Joint,
    dt: float,
    t_max: float,
    chunk_size: int | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Draw parameters from `prior` and simulate a two-condition CRDM dataset per draw.

    The test sampler for the CRDM recovery experiments. Each dataset is split down the
    middle, congruent first; both halves share one parameter draw, so congruency is a
    within-subject manipulation. See :func:`crdm_condition_design` for the ordering caveat.

    Args:
        key: PRNG key.
        batch_shape: ``(num_datasets, num_trials)``. Static.
        prior: A ``distrax.Joint`` over the seven CRDM parameters. Static.
        dt: Grid step. Static.
        t_max: Integration horizon. Static.
        chunk_size: Maximum elements per integration call. Static.

    Returns:
        ``(data, context)`` with `data` of shape ``(num_datasets, num_trials, 3)`` — columns
        ``[rt, choice, condition]`` — and `context` of shape ``(num_datasets, 7)`` in natural
        space.
    """
    key_context, key_data = jax.random.split(key, 2)

    num_datasets, num_trials = batch_shape
    context = jnp.stack(prior.sample(seed=key_context, sample_shape=(num_datasets,)), axis=-1)

    data = jax.vmap(simulate_crdm, in_axes=(0, 0, None, None, None, None))(
        jax.random.split(key_data, num_datasets),
        jnp.log(context), num_trials, dt, t_max, chunk_size,
    )
    return data, context


# --------------------------------------------------------------------------------------- #
# Hierarchical test samplers: one population per call.
# --------------------------------------------------------------------------------------- #


def sample_conditional_rdm_hierarchical_lkj_mvn(
    key, num_trials, num_subjects, **prior_kwargs,
):
    """Draw one population from the hierarchical RDM prior and simulate data for it.

    Args:
        key: PRNG key.
        num_trials: Trials per subject.
        num_subjects: Subjects in the population.
        **prior_kwargs: Forwarded to
            :func:`confrdm_jax.priors.create_hierarchical_rdm_prior_lkj_mvn`; supplied from
            ``conf_jax/model/rdm.yaml`` under ``hierarchical.prior``.

    Returns:
        ``(data, context)``. `data` has shape ``(S, num_trials, 2)`` with columns
        ``[rt, choice]``; `context` is the raw prior draw — a dict with keys ``s``, ``mu``,
        ``psi_raw``, ``z``, ``theta_bt`` — from which subject log-parameters are rebuilt with
        :func:`eamax.hierarchical.reconstruct_from_dict`.
    """
    key_context, key_data = jax.random.split(key)

    prior = create_hierarchical_rdm_prior_lkj_mvn(num_subjects, **prior_kwargs)
    context = prior.sample(seed=key_context)
    log_theta = reconstruct_from_dict(context)

    data = jax.vmap(simulate_rdm, in_axes=(0, 0, None))(
        jax.random.split(key_data, num_subjects), log_theta, num_trials
    )
    return data, context


def sample_conditional_crdm_hierarchical_lkj_mvn(
    key, num_trials, num_subjects, *, dt=0.001, t_max=4.0, chunk_size=None, **prior_kwargs,
):
    """Draw one population from the hierarchical CRDM prior and simulate data for it.

    Half the trials are congruent and half incongruent, congruent first; trial order is not
    randomised.

    Args:
        key: PRNG key.
        num_trials: Trials per subject.
        num_subjects: Subjects in the population.
        dt: Grid step.
        t_max: Integration horizon. Trials with no crossing carry ``rt = -1.0``.
        chunk_size: Maximum elements per integration call, to bound peak memory.
        **prior_kwargs: Forwarded to
            :func:`confrdm_jax.priors.create_hierarchical_crdm_prior_lkj_mvn`.

    Returns:
        ``(data, context)``. `data` has shape ``(S, num_trials, 3)`` with columns
        ``[rt, choice, condition]``; `context` is the raw prior draw.
    """
    key_context, key_data = jax.random.split(key)

    prior = create_hierarchical_crdm_prior_lkj_mvn(num_subjects, **prior_kwargs)
    context = prior.sample(seed=key_context)
    log_theta = reconstruct_from_dict(context)

    data = jax.vmap(simulate_crdm, in_axes=(0, 0, None, None, None, None))(
        jax.random.split(key_data, num_subjects), log_theta, num_trials, dt, t_max, chunk_size,
    )
    return data, context
