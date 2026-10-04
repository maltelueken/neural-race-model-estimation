"""Race likelihoods for the RDM and CRDM, single-subject and hierarchical.

Every likelihood here is :func:`eamax.race.race_loglik` with a different accumulator behind
it. The race itself — winner's density times losers' survival, the ``t0`` shift, the
likelihood floor, NaN containment, the padding mask and right-censoring — lives in `eamax`
and is applied once, to the assembled per-trial total. The one thing this module clamps is
a flow's *inputs*, to the box the flow was trained on, when a factory is given
``context_bounds`` — see :func:`_flow_accumulator`.

Which accumulator:

===============================  ==================================================
Model                            Accumulator
===============================  ==================================================
RDM, analytic                    :class:`eamax.accumulators.Wald` on both
RDM, neural                      :class:`confrdm_jax.flows_affine.FlowAccumulator` on both
CRDM, neural                     flow on the pulsed accumulator, ``Wald`` on the other
CRDM, reference                  :func:`eamax.accumulators.solve_volterra_fpt`, once per
                                 condition, on the pulsed accumulator; ``Wald`` on the other
===============================  ==================================================

**The hybrid race.** Only one accumulator per CRDM trial carries the conflict pulse, and
only that one needs the expensive density. :func:`eamax.race.gather_by_mask` collapses the
``(N, T)`` parameters to the ``(T,)`` the pulsed accumulator sees, so it is evaluated **once
per trial rather than once per accumulator**, and :func:`~eamax.race.overlay_by_mask` puts
the result back. Which accumulator is pulsed comes from the design's ``distractor`` column —
see :mod:`confrdm_jax.specs`.

**Parameters arrive in log space.** MCMC samples ``log theta``; the spec's ``constrain``
exponentiates on entry and the matching Jacobian belongs to the prior (see
:func:`confrdm_jax.priors.log_prior_fn`).

**Censoring.** The simulators emit ``rt = -1.0`` for trials that never crossed within
``t_max``, which is the observation ``T > t_max``. Passing ``t_max`` to a CRDM factory
scores those by the product of every accumulator's survival at ``t_max`` — the race
analogue of the right-censored MLE :func:`confrdm_jax.flows_affine.loss_fn` applies on the
training side.
Leave it ``None`` only for data that cannot contain the sentinel. A sentinel that reaches a
``t_max=None`` likelihood is not an error and not floored either: the trial has no winner, so
it is scored as "every accumulator survived" — but at the clamped decision time, where that
probability is ~1, so it contributes ~0 and is silently ignored. Hence the
``t_max: ${model.test_sampler.t_max}`` interpolation in ``conf_jax/model/crdm.yaml``, which
makes the likelihood's horizon and the simulator's impossible to set separately.

**No ``rt <= t0`` penalty.** `eamax` evaluates such a trial at the clamped decision time and
lets it land on the same flat floor as any other hopeless trial, so the likelihood says
nothing about which way ``t0`` should move. That region is kept out of by construction, at
initialisation, with :class:`eamax.inference.init.T0Support`.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
from eamax.accumulators import VolterraPulsedWald, Wald, solve_volterra_fpt
from eamax.accumulators.pulse import time_grid
from eamax.design import build_params_fn
from eamax.numerics import guard_positive
from eamax.race import gather_by_mask, overlay_by_mask, race_loglik, winner_mask

from .flows_affine import FlowAccumulator
from .specs import CRDM_CONTEXT_NAMES, WALD_CONTEXT_NAMES, crdm_spec, make_design, rdm_spec

#: Density and survival floor before the log, as in :class:`eamax.accumulators.VolterraPulsedWald`.
_VOLTERRA_TINY = 1e-30

__all__ = [
    "create_crdm_hierarchical_likelihood_factory_approx",
    "create_crdm_likelihood_factory_approx",
    "create_crdm_likelihood_volterra",
    "create_rdm_hierarchical_likelihood",
    "create_rdm_hierarchical_likelihood_factory_approx",
    "create_rdm_likelihood_factory_approx",
    "create_rdm_two_accumulators_likelihood",
]


def _normalize_context_bounds(context_bounds, context_names):
    """`context_bounds` as ``{name: (low, high)}`` floats, checked against `context_names`.

    Accepts any mapping of name to a two-element sequence, so an OmegaConf ``DictConfig``
    handed over by Hydra works as-is. A name the flow is not conditioned on is an error
    rather than ignored: it is almost always a typo, and a silently absent clamp is exactly
    the failure the clamp exists to prevent.
    """
    if context_bounds is None:
        return {}
    bounds = {}
    for key, limits in context_bounds.items():
        name = str(key)
        if name not in context_names:
            raise ValueError(
                f"context_bounds names {name!r}, which is not a flow input; "
                f"expected a subset of {list(context_names)}.",
            )
        low, high = (float(limit) for limit in limits)
        if not low < high:
            raise ValueError(f"context_bounds[{name!r}] = ({low}, {high}) is empty.")
        bounds[name] = (low, high)
    return bounds


def _clamped(transform, low, high):
    """`transform` (or the identity) followed by a clip to ``[low, high]``."""
    if transform is None:
        return lambda value: jnp.clip(value, low, high)
    return lambda value: jnp.clip(transform(value), low, high)


def _flow_accumulator(conditioner, context_names, remat=False, context_bounds=None):
    """A flow accumulator over `context_names`.

    :class:`confrdm_jax.flows_affine.FlowAccumulator`, which is `eamax`'s for a plain
    conditioner and adds the per-context affine stage for one trained with
    ``model.flow_affine=true``; the layout is read off the conditioner's weights.

    ``amp`` is passed through ``abs`` as a guard: the flow was trained on non-negative
    amplitudes, and under :mod:`confrdm_jax.specs` ``amp`` is already non-negative, so the
    transform is a no-op on current configurations.

    **Clamping to the training box.** With `context_bounds`, each named input is clipped to
    the ``(low, high)`` range the conditioner was trained over before it reaches the flow.
    Outside that box the spline flow extrapolates, and its log-density develops gradient
    spikes and flat plateaus that collapse step-size adaptation and freeze an SMC cloud.
    Clipped, a particle
    that strays outside sees the density at the box edge instead: bounded and continuous,
    with zero gradient along the clipped direction, so the prior alone pulls it back. Inside
    the box the likelihood is bit-identical. Only the flow's inputs are clipped; the inverse
    Gaussian accumulator of the hybrid race stays exact everywhere.
    """
    transform = {"amp": jnp.abs} if "amp" in context_names else {}
    for name, (low, high) in _normalize_context_bounds(context_bounds, context_names).items():
        transform[name] = _clamped(transform.get(name), low, high)
    transform = transform or None
    # `FlowAccumulator` defaults to float32 on the grounds that flows usually train there.
    # This repository trains and infers under `jax_enable_x64`, so the conditioner's weights
    # are float64 and casting the context down would discard ~1e-6 of density per trial for
    # nothing. `result_type(float)` is float64 with x64 on and float32 without, so the
    # accumulator follows whatever precision the run was configured for.
    return FlowAccumulator(
        conditioner, context_names, transform=transform,
        dtype=jnp.result_type(float), remat=remat,
    )


def _race(spec, accumulator, theta, design, t_max=None):
    """Per-trial log-likelihood of one dataset under `spec` and `accumulator`."""
    params, t0 = build_params_fn(spec, accumulator)(theta, design)
    return race_loglik(
        design.rt,
        design.response,
        t0,
        lambda t: accumulator.log_pdf_sf(t, params),
        mask=design.mask,
        t_max=t_max,
        first_response=spec.first_response,
    )


def _hybrid_race(spec, pulsed, plain, theta, design, t_max=None):
    """Per-trial log-likelihood with `pulsed` on the distractor accumulator and `plain` elsewhere.

    `pulsed` is evaluated once per trial, on the gathered ``(T,)`` parameters of whichever
    accumulator the design's ``distractor`` column names; `plain` covers all ``N``.
    """
    params, t0 = build_params_fn(spec, pulsed)(theta, design)
    pulse_mask = winner_mask(design.distractor, spec.num_responses, spec.first_response)
    gathered = dict(
        zip(pulsed.param_names, gather_by_mask(pulse_mask, *(params[n] for n in pulsed.param_names)))
    )
    plain_params = {name: params[name] for name in plain.param_names}

    def pdf_sf_fn(t):
        plain_pdf, plain_sf = plain.log_pdf_sf(t, plain_params)
        pulsed_pdf, pulsed_sf = pulsed.log_pdf_sf(t, gathered)
        return (
            overlay_by_mask(pulse_mask, pulsed_pdf, plain_pdf),
            overlay_by_mask(pulse_mask, pulsed_sf, plain_sf),
        )

    return race_loglik(
        design.rt,
        design.response,
        t0,
        pdf_sf_fn,
        mask=design.mask,
        t_max=t_max,
        first_response=spec.first_response,
    )


def _hierarchical(race_fn, data, mask):
    """Lift a per-subject per-trial log-likelihood to a scalar total over subjects.

    `mask` exists so subjects with differing trial counts can be padded to a rectangular
    array; masked trials contribute exactly 0 rather than being dropped, which keeps shapes
    static under ``jit``. Population-level parameters never appear here — the caller
    reconstructs ``(S, P)`` subject log-parameters from them first (see
    :meth:`eamax.hierarchical.HierarchicalFlatSpace.subject_params`).
    """
    per_subject = jax.vmap(race_fn, in_axes=(0, 0, 0))

    def likelihood_fun(theta):
        return jnp.sum(per_subject(theta, data, mask))

    return likelihood_fun


# --------------------------------------------------------------------------------------- #
# RDM
# --------------------------------------------------------------------------------------- #


def create_rdm_two_accumulators_likelihood(data):
    """Reference RDM likelihood, using the analytic inverse Gaussian.

    Args:
        data: One dataset, ``(T, 2)`` with columns ``[rt, choice]``.

    Returns:
        ``likelihood_fun(log_theta)`` taking a ``(5,)`` log-parameter vector and returning
        per-trial log-likelihoods of shape ``(T,)``. Callers sum.
    """
    spec = rdm_spec()
    design = make_design(data)
    accumulator = Wald()

    @jax.jit
    def likelihood_fun(theta):
        return _race(spec, accumulator, theta, design)

    return likelihood_fun


def create_rdm_likelihood_factory_approx(conditioner, context_bounds=None):
    """Factory for the neural-approximate RDM likelihood.

    Args:
        conditioner: Trained conditioner, conditioned on
            :data:`confrdm_jax.specs.WALD_CONTEXT_NAMES`.
        context_bounds: Optional ``{name: (low, high)}`` training box; flow inputs are
            clipped to it. See :func:`_flow_accumulator`. Wired from the training prior in
            ``conf_jax/model/rdm.yaml``.

    Returns:
        ``create_likelihood(data) -> likelihood_fun(log_theta) -> (T,)``, with the same
        signature as :func:`create_rdm_two_accumulators_likelihood`'s result.
    """
    spec = rdm_spec()
    accumulator = _flow_accumulator(
        conditioner, WALD_CONTEXT_NAMES, context_bounds=context_bounds,
    )

    def create_likelihood(data):
        design = make_design(data)

        def likelihood_fun(theta):
            return _race(spec, accumulator, theta, design)

        return likelihood_fun

    return create_likelihood


def create_rdm_hierarchical_likelihood(data, mask):
    """Hierarchical analytic RDM likelihood.

    Args:
        data: Padded trial data, shape ``(S, T, 2)`` with columns ``[rt, choice]``.
        mask: Boolean, shape ``(S, T)``. True for real trials.

    Returns:
        ``likelihood_fun(log_theta)`` taking subject-level log-parameters of shape ``(S, 5)``
        and returning a scalar total log-likelihood across subjects and trials.
    """
    spec = rdm_spec()
    accumulator = Wald()

    def one_subject(theta, subject_data, subject_mask):
        return _race(spec, accumulator, theta, make_design(subject_data, subject_mask))

    return jax.jit(_hierarchical(one_subject, data, mask))


def create_rdm_hierarchical_likelihood_factory_approx(
    conditioner, remat=True, context_bounds=None,
):
    """Factory for the hierarchical neural-approximate RDM likelihood.

    Args:
        conditioner: Trained conditioner over :data:`confrdm_jax.specs.WALD_CONTEXT_NAMES`.
        remat: Recompute the flow's forward pass during the backward pass instead of storing
            its activations. Trades time for memory, which is the binding constraint once the
            flow is evaluated for every subject of a population at once.
        context_bounds: See :func:`create_rdm_likelihood_factory_approx`.

    Returns:
        ``create_likelihood(data, mask)`` returning a function with the same signature as
        :func:`create_rdm_hierarchical_likelihood`'s.
    """
    spec = rdm_spec()
    accumulator = _flow_accumulator(
        conditioner, WALD_CONTEXT_NAMES, remat=remat, context_bounds=context_bounds,
    )

    def create_likelihood(data, mask):
        def one_subject(theta, subject_data, subject_mask):
            return _race(spec, accumulator, theta, make_design(subject_data, subject_mask))

        return _hierarchical(one_subject, data, mask)

    return create_likelihood


# --------------------------------------------------------------------------------------- #
# CRDM
# --------------------------------------------------------------------------------------- #


def create_crdm_likelihood_factory_approx(conditioner, t_max=None, context_bounds=None):
    """Factory for the hybrid neural/analytic CRDM likelihood.

    Args:
        conditioner: Trained conditioner over :data:`confrdm_jax.specs.CRDM_CONTEXT_NAMES`.
        t_max: Integration horizon of the simulator that produced the data, on the
            **decision-time** scale. Trials carrying the ``rt = -1.0`` censoring sentinel are
            then scored by the product of both accumulators' survival at ``t_max``. Wired
            from ``model.test_sampler.t_max`` in ``conf_jax/model/crdm.yaml``, so the
            likelihood's horizon and the simulator's cannot drift apart.
        context_bounds: Optional ``{name: (low, high)}`` training box; the pulsed
            accumulator's flow inputs are clipped to it. See :func:`_flow_accumulator`.
            Wired from the training prior in ``conf_jax/model/crdm.yaml``.

    Returns:
        ``create_likelihood(data) -> likelihood_fun(log_theta) -> (T,)``. `data` is
        ``(T, 3)`` with columns ``[rt, choice, condition]``, condition being 1 for congruent
        and 0 for incongruent.
    """
    spec = crdm_spec()
    pulsed = _flow_accumulator(conditioner, CRDM_CONTEXT_NAMES, context_bounds=context_bounds)
    plain = Wald()

    def create_likelihood(data):
        design = make_design(data)

        def likelihood_fun(theta):
            return _hybrid_race(spec, pulsed, plain, theta, design, t_max=t_max)

        return likelihood_fun

    return create_likelihood


def create_crdm_hierarchical_likelihood_factory_approx(
    conditioner, t_max=None, remat=True, context_bounds=None,
):
    """Factory for the hierarchical hybrid CRDM likelihood.

    Censoring matters more here than in the single-subject case: ``t0`` is per subject, so an
    unhandled sentinel would drag only its own subject's ``t0`` with nothing pooling the
    damage away.

    Args:
        conditioner: Trained conditioner over :data:`confrdm_jax.specs.CRDM_CONTEXT_NAMES`.
        t_max: See :func:`create_crdm_likelihood_factory_approx`. Wired from
            ``model.hierarchical.test_sampler.t_max``.
        remat: See :func:`create_rdm_hierarchical_likelihood_factory_approx`.
        context_bounds: See :func:`create_crdm_likelihood_factory_approx`. This is where the
            clamp matters most: tempered SMC evaluates the likelihood at every particle drawn
            from the prior, tails included.

    Returns:
        ``create_likelihood(data, mask)`` returning ``likelihood_fun(log_theta)`` over
        ``(S, 7)`` subject log-parameters, for `data` of shape ``(S, T, 3)``.
    """
    spec = crdm_spec()
    pulsed = _flow_accumulator(
        conditioner, CRDM_CONTEXT_NAMES, remat=remat, context_bounds=context_bounds,
    )
    plain = Wald()

    def create_likelihood(data, mask):
        def one_subject(theta, subject_data, subject_mask):
            return _hybrid_race(
                spec, pulsed, plain, theta,
                make_design(subject_data, subject_mask), t_max=t_max,
            )

        return _hierarchical(one_subject, data, mask)

    return create_likelihood


def _volterra_horizon(data, mask, censor_t_max):
    """The shortest solver horizon that covers every decision time `data` can produce.

    The Fortet recursion is causal -- the density at ``t`` depends only on the grid before
    it -- so stopping the grid at the data's last decision time changes no value the race
    reads, and the cost falls with the square of the grid length. A decision time is
    ``rt - t0 < rt``, so the longest observed RT bounds it for every ``t0`` the sampler can
    propose. A censored trial is evaluated at decision time `censor_t_max`, so when the
    sentinel is present and scored, the grid has to reach that instead.
    """
    try:
        rt = np.asarray(data[:, 0])
    except jax.errors.TracerArrayConversionError:
        raise TypeError(
            "create_crdm_likelihood_volterra derives its grid horizon from the data, which "
            "needs concrete data; under jit or vmap pass t_max explicitly.",
        ) from None
    valid = np.ones(rt.shape, bool) if mask is None else np.asarray(mask, bool)
    horizon = float(np.max(rt[valid & (rt > 0)], initial=0.0))
    if censor_t_max is not None and np.any(valid & (rt < 0)):
        horizon = max(horizon, float(censor_t_max))
    if horizon <= 0:
        raise ValueError("No positive response times to set the solver horizon from.")
    return horizon


def create_crdm_likelihood_volterra(data, dt=0.001, t_max=None, censor_t_max=None, mask=None):
    """Reference CRDM likelihood: the Volterra solver in place of the flow.

    The same hybrid race as :func:`create_crdm_likelihood_factory_approx`, with the pulsed
    accumulator's density computed by deterministic numerical integration
    (:func:`eamax.accumulators.solve_volterra_fpt`) rather than learned. That is the
    quantity the flow approximates, so this is what a flow-versus-reference comparison is
    against.

    **One solve per condition, not per trial.** Under :func:`~confrdm_jax.specs.crdm_spec`
    the pulsed accumulator's parameters depend on the trial only through its condition: a
    congruent trial pulses the target (drift ``v_c_intercept + v_c_slope``, noise
    ``s_true``), an incongruent one the non-target (``v_c_intercept``, noise 1). So there
    are exactly two parameter sets per evaluation. Each is solved once and every trial is
    interpolated from its condition's grid -- the values are those of
    :class:`eamax.accumulators.VolterraPulsedWald`, which solves once per trial, at a
    fraction of the cost.

    **The grid stops at the data.** By default the solver horizon is the longest observed
    RT (or `censor_t_max`, when a censored trial needs it); see :func:`_volterra_horizon`.
    Cost is ``O(num_steps^2)`` per solve, so this matters more than anything else here.

    Differentiable, so it runs under NUTS; the gradient costs roughly 8x the value, from
    reverse mode through the recursion. Not reachable from Hydra: ``conf_jax/model/crdm.yaml``
    declares no ``likelihood_factory_ref`` and sets ``run_reference_recovery: false``.

    Args:
        data: One dataset, ``(T, 3)`` with columns ``[rt, choice, condition]``, condition
            coded 0 (incongruent) / 1 (congruent).
        dt: Solver grid step.
        t_max: Solver grid horizon. ``None`` derives it from `data`, which must then be
            concrete. An explicit value must cover the data: decision times beyond it are
            extrapolated by holding the endpoint.
        censor_t_max: Censoring horizon of the *simulator*, if its sentinel can appear.
            Distinct from `t_max`, which is the solver's grid.
        mask: Optional boolean ``(T,)``; False trials contribute exactly zero and do not
            count towards the horizon.

    Returns:
        ``likelihood_fun(log_theta) -> (T,)``.
    """
    if t_max is None:
        t_max = _volterra_horizon(data, mask, censor_t_max)
    num_steps = math.ceil(t_max / dt - 1e-9)

    spec = crdm_spec()
    design = make_design(data, mask)
    # Only for its parameter names, guards and pulse shape; the solve is done here, per cell.
    pulsed = VolterraPulsedWald(dt=dt, t_max=num_steps * dt)
    plain = Wald()
    grid = time_grid(dt, num_steps)

    # A two-trial design with one trial per condition: the spec maps it to exactly the two
    # parameter sets the pulsed accumulator can take.
    conditions = jnp.arange(2, dtype=design.rt.dtype)
    cell_design = make_design(jnp.stack([jnp.ones(2), jnp.ones(2), conditions], axis=1))
    cell_pulse_mask = winner_mask(cell_design.distractor, spec.num_responses, spec.first_response)
    trial_cell = design.condition.astype(jnp.int32)
    trial_index = jnp.arange(trial_cell.shape[0])

    def solve(v, amp, tau, s, b):
        return solve_volterra_fpt(v, amp, tau, s, b, dt, num_steps, pulsed.a_shape)

    def likelihood_fun(theta):
        params, t0 = build_params_fn(spec, pulsed)(theta, design)
        plain_params = {name: params[name] for name in plain.param_names}
        pulse_mask = winner_mask(design.distractor, spec.num_responses, spec.first_response)

        cell_params, _ = build_params_fn(spec, pulsed)(theta, cell_design)
        cell = dict(
            zip(
                pulsed.param_names,
                gather_by_mask(cell_pulse_mask, *(cell_params[n] for n in pulsed.param_names)),
            )
        )
        for name in ("tau", "s", "b"):
            cell[name] = guard_positive(cell[name], pulsed.min_param)
        g, cdf = jax.vmap(solve)(*(cell[n] for n in pulsed.param_names))  # (2, num_steps)

        def pdf_sf_fn(t):
            # Every trial against both grids, then each keeps its own condition's: two
            # interpolations of (T,) points, rather than a (T, num_steps) gather.
            def at_trials(curve):
                values = jax.vmap(lambda row: jnp.interp(t, grid, row))(curve)
                return values[trial_cell, trial_index]

            pulsed_pdf = jnp.log(jnp.maximum(at_trials(g), _VOLTERRA_TINY))
            pulsed_sf = jnp.log(jnp.maximum(1.0 - at_trials(cdf), _VOLTERRA_TINY))
            plain_pdf, plain_sf = plain.log_pdf_sf(t, plain_params)
            return (
                overlay_by_mask(pulse_mask, pulsed_pdf, plain_pdf),
                overlay_by_mask(pulse_mask, pulsed_sf, plain_sf),
            )

        return race_loglik(
            design.rt,
            design.response,
            t0,
            pdf_sf_fn,
            mask=design.mask,
            t_max=censor_t_max,
            first_response=spec.first_response,
        )

    return likelihood_fun
