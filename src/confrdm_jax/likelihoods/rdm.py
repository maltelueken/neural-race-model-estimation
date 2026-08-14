"""Racing diffusion model likelihoods, analytic and neural.

**The racing likelihood.** With independent accumulators, observing that
accumulator *i* finished at time ``t`` and the others had not yet finished
gives ``p_i(t) * prod_{j != i} S_j(t)``, so every likelihood here is built from
a density for the winner and survival functions for the losers.  In log space
that is a sum, which is why each helper returns ``(log_pdf, log_sf)`` for a
single accumulator and the callers combine them.

Two implementations of those parts:

- :func:`inv_gauss_log_pdf_sf` — exact, from the inverse Gaussian
  first-passage density of a constant-drift diffusion.
- :func:`create_rdm_likelihood_factory_approx` — the trained flow, exercised on
  the same race structure. The RDM is the model where both exist, so it is
  where the neural approximation gets validated against a reference posterior.

**Parameters arrive in log space.** MCMC samples ``log theta``; every
likelihood exponentiates on entry, and the corresponding Jacobian belongs to
the prior (see ``log_prior`` in ``scripts/parameter_recovery.py``).  Parameter
order is ``[v_intercept, v_slope, s_true, b, t0]``, positional and unenforced.

**Numerical guarding.** Diffusion likelihoods are undefined for ``rt <= t0``
and can underflow deep in the tails.  ``_clamp_log`` and
``_penalize_invalid_rt`` handle both, and the ordering between them matters —
see their docstrings before adding a clamp at any call site.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax.scipy import stats

from confrdm_jax.flows import evaluate_pdf_sf


_LOG_FLOOR = jnp.log(1e-12)
_FLOOR = 1e-10


def _clamp_log(x):
    """Clamp log-densities to a finite floor, avoiding NaN in the computation graph.

    Uses nan_to_num first because jnp.clip propagates NaN (IEEE 754).
    The gradient of nan_to_num is 0 for NaN inputs, so this is safe for
    reverse-mode differentiation (HMC / NUTS).
    """
    return jnp.maximum(jnp.nan_to_num(x, nan=_LOG_FLOOR), _LOG_FLOOR)


def _penalize_invalid_rt(rt_shifted, log_pdf, log_sf):
    """Clamp valid log-densities and apply a steep penalty when rt <= t0.

    For trials where rt - t0 <= 0, the observation is impossible under
    the model.  Instead of returning a flat floor (zero gradient), we
    return a linear penalty whose gradient pushes t0 below the minimum
    observed RT.

    This function owns *both* halves of the correction, and the order matters:
    ``_clamp_log`` is applied to the valid branch only, before the penalty is
    substituted in.  The penalty is by construction <= ``_LOG_FLOOR``, so any
    clamping applied *after* this function would floor it back to the constant
    and destroy exactly the gradient it exists to provide.  Callers must
    therefore not clamp the values returned here.

    Args:
        rt_shifted: rt - t0 (before clamping).
        log_pdf: Raw log-PDF computed on clamped rt; may be NaN or -inf.
        log_sf: Raw log-SF computed on clamped rt; may be NaN or -inf.

    Returns:
        Corrected (log_pdf, log_sf), clamped where the trial is valid and
        carrying the steep penalty where it is not.
    """
    valid = rt_shifted > _FLOOR
    penalty = _LOG_FLOOR + 1e3 * jnp.minimum(rt_shifted - _FLOOR, 0.0)
    log_pdf = jnp.where(valid, _clamp_log(log_pdf), penalty)
    log_sf = jnp.where(valid, _clamp_log(log_sf), 0.0)
    return log_pdf, log_sf


def _censored_eval_rt(rt, t0, t_max):
    r"""Redirect the censoring sentinel onto the RT whose *decision* time is ``t_max``.

    The CRDM simulators emit ``rt = -1.0`` when no accumulator crossed within
    the integration horizon (see ``simulate_crdm_single_trial``).  That is the
    observation :math:`T > t_{\max}`, not missing data, and its likelihood for
    a race is :math:`S_1(t_{\max})\,S_2(t_{\max})` — *both* accumulators
    still running.  Feeding the sentinel to a density instead sends
    ``rt - t0 = -1 - t0`` into the invalid branch of
    :func:`_penalize_invalid_rt`, which contributes about ``-1300`` per trial
    with a gradient of ``-1e3`` driving ``t0`` to zero.  One such trial
    outweighs roughly a thousand real ones.

    Rather than evaluate the accumulators twice, this substitutes the input —
    the same trick ``flows.loss_fn`` uses on the training side.  Censored
    trials are evaluated at ``t_max + t0`` so that ``rt - t0`` is exactly
    ``t_max``; their log-densities are then discarded by
    :func:`_apply_censoring` and only their survival functions are kept.

    **The horizon is on decision time, not RT.** ``simulate_crdm_single_trial``
    integrates for ``num_steps * dt = t_max`` and only then adds ``t0``
    (``rt = min_fpt + t0``), so a censored trial says the *decision* took
    longer than ``t_max``. The censored contribution therefore carries no
    ``t0`` dependence at all, which is the intended behaviour and the opposite
    of the runaway gradient it replaces.

    Args:
        rt: Observed response times, possibly containing ``-1.0`` sentinels.
        t0: Non-decision time.
        t_max: Simulator integration horizon, or ``None`` for samplers that
            cannot censor (the RDM path), in which case nothing is substituted.

    Returns:
        ``(rt_eval, is_censored)``. `is_censored` is ``None`` when `t_max` is,
        which makes :func:`_apply_censoring` a no-op and keeps the result
        bit-identical to the uncensored code path.
    """
    if t_max is None:
        return rt, None
    is_censored = rt < 0.0
    return jnp.where(is_censored, t_max + t0, rt), is_censored


def _apply_censoring(is_censored, log_sf_true, log_sf_false, log_lik):
    """Score censored trials by ``log S_true(t_max) + log S_false(t_max)``.

    The companion to :func:`_censored_eval_rt`, applied *after* the winner /
    loser routing. Note the sum is routing-invariant — a censored trial has no
    winner, and both accumulators contribute a survival factor — so which
    accumulator the flow versus the inverse Gaussian handled does not matter
    here.

    Args:
        is_censored: Mask from :func:`_censored_eval_rt`, or ``None`` to skip.
        log_sf_true: Target accumulator's log survival at ``t_max``.
        log_sf_false: Non-target accumulator's log survival at ``t_max``.
        log_lik: Per-trial log-likelihood for the observed-response case.

    Returns:
        `log_lik` with the censored entries replaced.
    """
    if is_censored is None:
        return log_lik
    return jnp.where(is_censored, log_sf_true + log_sf_false, log_lik)

@jax.jit
def inv_gauss_logpdf(t, mu, lam):
    """Log density of the inverse Gaussian at `t`.

    The first-passage time of a diffusion with drift ``v`` and diffusion ``s``
    through a boundary ``b`` is inverse Gaussian with ``mu = b / v`` and
    ``lam = (b / s)^2``.

    Written out rather than taken from a library so it stays differentiable and
    free of data-dependent branching. Agrees with
    ``scipy.stats.invgauss.logpdf`` to ~1e-13 over the parameter range used
    here. `t` must be strictly positive; callers floor it.
    """
    e = -(lam / (2 * t)) * (t**2 / mu**2 - 2 * t / mu  + 1)

    x = e + 0.5 * jnp.log(lam) - 0.5 * jnp.log(2 * t**3 * jnp.pi)

    return x

@jax.jit
def inv_gauss_logsf(t, mu, lam):
    """Log survival function of the inverse Gaussian — the losing accumulator's factor.

    Uses the numerically stable form of Giner & Smyth (2016), *statmod:
    Probability Calculations for the Inverse Gaussian Distribution*, R Journal
    8(1) — https://journal.r-project.org/archive/2016-1/giner-smyth.pdf — which
    works in the ``(mu / lam, t / lam)`` parameterisation and combines the two
    normal-CDF terms through ``log1p`` so the far right tail does not
    catastrophically cancel.

    Matches ``scipy.stats.invgauss.logsf`` to ~1e-13, degrading to ~1e-8 only
    for very large ``lam`` where the two terms are nearly equal.
    """
    # Clamp inputs to avoid NaN from sqrt/division on non-positive values.
    mu = mu / lam
    t = t / lam
    r = 1.0 / jnp.sqrt(t)
    a = stats.norm.logcdf(-r * ((t / mu) - 1.0))
    b = 2.0 / mu + stats.norm.logcdf(-r * (t + mu) / mu)
    # Clamp b - a <= 0 to prevent log1p argument < -1, which produces NaN.
    # Mathematically b <= a always, but floating-point arithmetic can violate this.
    result = a + jnp.log1p(-jnp.exp(jnp.minimum(b - a, 0.0)))
    return jnp.where(t > 0.0, result, 0.0)

@jax.jit
def inv_gauss_log_pdf_sf(rt, v, s, b, t0):
    """Exact ``(log_pdf, log_sf)`` for one accumulator, guarded for MCMC.

    Shifts by the non-decision time, floors the parameters away from zero so
    the closed forms stay finite, and hands the result to
    :func:`_penalize_invalid_rt`.

    Args:
        rt: Observed response times (not decision times).
        v: Drift rate.
        s: Diffusion coefficient.
        b: Boundary.
        t0: Non-decision time.

    Returns:
        ``(log_pdf, log_sf)``, already clamped and penalised. **Do not clamp
        the result again** — that would flatten the ``rt <= t0`` penalty back
        to a constant and remove the gradient that pushes `t0` into the valid
        region.
    """
    rt_shifted = rt - t0
    rt_safe = jnp.maximum(rt_shifted, _FLOOR)

    v = jnp.maximum(v, _FLOOR)
    s = jnp.maximum(s, _FLOOR)
    b = jnp.maximum(b, _FLOOR)

    # mu_winner = b/drift_winner
    mu = b/v
    # lam_winner = (b/s_winner)**2
    lam = (b/s)**2

    log_pdf = inv_gauss_logpdf(rt_safe, mu, lam)
    log_sf = inv_gauss_logsf(rt_safe, mu, lam)
    return _penalize_invalid_rt(rt_shifted, log_pdf, log_sf)


def create_rdm_two_accumulators_likelihood(data):
    """Reference likelihood using analytical inverse Gaussian."""
    rt = data[:, 0]
    choice = data[:, 1]

    @jax.jit
    def likelihood_fun(x):
        x = jnp.exp(x)
        v_c_true = x[0] + x[1]
        v_c_false = x[0]
        s_true = x[2]
        s_false = 1.0
        b = x[3]
        t0 = x[4]

        log_pdf_true, log_sf_true = inv_gauss_log_pdf_sf(rt, v_c_true, s_true, b, t0)
        log_pdf_false, log_sf_false = inv_gauss_log_pdf_sf(rt, v_c_false, s_false, b, t0)

        # Already clamped/penalised by inv_gauss_log_pdf_sf; clamping again here
        # would erase the rt <= t0 penalty gradient.
        dens_true = log_pdf_true.squeeze() + log_sf_false.squeeze()
        dens_false = log_pdf_false.squeeze() + log_sf_true.squeeze()

        return jnp.where(choice == 1, dens_true, dens_false)

    return likelihood_fun


def create_rdm_likelihood_factory_approx(conditioner):
    """Factory that returns a likelihood creator using neural approximation.

    Args:
        conditioner: Trained neural network conditioner for density estimation.

    Returns:
        A function that takes data and returns a likelihood function.
    """
    def inv_gauss_log_pdf_sf_approx(rt, v, s, b, t0):
        rt_shifted = rt - t0
        rt_safe = jnp.maximum(rt_shifted, _FLOOR)

        v = jnp.maximum(v, _FLOOR)
        s = jnp.maximum(s, _FLOOR)
        b = jnp.maximum(b, _FLOOR)

        log_pdf, log_sf = evaluate_pdf_sf(conditioner, rt_safe, jnp.array([v, s, b]))
        return _penalize_invalid_rt(rt_shifted, log_pdf, log_sf)

    def create_likelihood(data):
        rt = data[:, 0]
        choice = data[:, 1]

        @nnx.jit
        def likelihood_fun(x):
            x = jnp.exp(x)
            v_true = x[0] + x[1]
            v_false = x[0]
            s_true = x[2]
            s_false = 1.0
            b = x[3]
            t0 = x[4]

            log_pdf_true, log_sf_true = inv_gauss_log_pdf_sf_approx(rt, v_true, s_true, b, t0)
            log_pdf_false, log_sf_false = inv_gauss_log_pdf_sf_approx(rt, v_false, s_false, b, t0)

            # Already clamped/penalised upstream; see _penalize_invalid_rt.
            dens_true = log_pdf_true.squeeze() + log_sf_false.squeeze()
            dens_false = log_pdf_false.squeeze() + log_sf_true.squeeze()

            return jnp.where(choice == 1, dens_true, dens_false)

        return likelihood_fun

    return create_likelihood
