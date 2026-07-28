"""Hierarchical racing diffusion model likelihood.

The per-trial density is exactly the single-subject one; what changes is the
interface.  These factories take ``(data, mask)`` up front and return a
function of subject-level log-parameters shaped ``(S, P)`` — not a flat vector
— which is then ``vmap``ped over subjects and summed to a scalar.

Population-level parameters never appear here.  The caller reconstructs
``log_theta`` from ``(mu, s, psi_raw, z, theta_bt)`` and passes only the result,
so the hierarchical structure lives entirely in the prior and in
``scripts/parameter_recovery_hierarchical.py``.

`mask` exists so subjects with differing trial counts can be padded to a
rectangular array; masked trials contribute exactly 0 rather than being
dropped, which keeps shapes static under ``jit``.  The current recovery script
gives every subject the same number of trials and passes an all-true mask.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from confrdm_jax.flows import evaluate_pdf_sf
from .rdm import _penalize_invalid_rt
from .rdm import inv_gauss_log_pdf_sf


_FLOOR = 1e-10


def create_rdm_hierarchical_likelihood(data, mask):
    """Hierarchical analytical likelihood for multi-subject RDM.

    Args:
        data: Padded trial data, shape (S, max_trials, 2) with columns [RT, choice].
        mask: Boolean mask, shape (S, max_trials). True for real trials.

    Returns:
        A JIT-compiled function `likelihood_fun(log_theta)` taking subject-level
        log-parameters of shape (S, 5) — columns
        ``[v_intercept, v_slope, s_true, b, t0]`` — and returning a scalar total
        log-likelihood across all subjects and trials. Population-level
        parameters do not enter here; the caller reconstructs `log_theta` from
        them first (see `scripts/parameter_recovery_hierarchical.py`).
    """

    def _single_subject_ll(subject_data, subject_mask, subject_params):
        """Compute per-trial log-likelihoods for one subject."""
        rt = subject_data[:, 0]
        choice = subject_data[:, 1]

        x = jnp.exp(subject_params)
        v_c_true = x[0] + x[1]
        v_c_false = x[0]
        s_true = x[2]
        s_false = 1.0
        b = x[3]
        t0 = x[4]

        log_pdf_true, log_sf_true = inv_gauss_log_pdf_sf(rt, v_c_true, s_true, b, t0)
        log_pdf_false, log_sf_false = inv_gauss_log_pdf_sf(rt, v_c_false, s_false, b, t0)

        # Already clamped/penalised upstream; see _penalize_invalid_rt.
        dens_true = log_pdf_true.squeeze() + log_sf_false.squeeze()
        dens_false = log_pdf_false.squeeze() + log_sf_true.squeeze()

        ll = jnp.where(choice == 1, dens_true, dens_false)

        return jnp.sum(jnp.where(subject_mask, ll, 0.0))

    _vmapped_ll = jax.vmap(_single_subject_ll, in_axes=(0, 0, 0))

    @jax.jit
    def likelihood_fun(x):
        return jnp.sum(_vmapped_ll(data, mask, x))

    return likelihood_fun


def create_rdm_hierarchical_likelihood_factory_approx(conditioner):
    """Factory for hierarchical neural-approximate likelihood for multi-subject RDM.

    Args:
        conditioner: Trained neural network conditioner for density estimation.

    Returns:
        A function `create_likelihood(data, mask)` returning a likelihood
        function with the same signature as
        `create_rdm_hierarchical_likelihood`'s: it takes subject-level
        log-parameters of shape (S, 5), not a flat vector.
    """
    def inv_gauss_log_pdf_sf_approx(rt, v, s, b, t0):
        rt_shifted = rt - t0
        rt_safe = jnp.maximum(rt_shifted, _FLOOR)

        v = jnp.maximum(v, _FLOOR)
        s = jnp.maximum(s, _FLOOR)
        b = jnp.maximum(b, _FLOOR)

        log_pdf, log_sf = evaluate_pdf_sf(conditioner, rt_safe, jnp.array([v, s, b]))
        return _penalize_invalid_rt(rt_shifted, log_pdf, log_sf)

    def create_likelihood(data, mask):
        def _single_subject_ll(subject_data, subject_mask, subject_params):
            """Compute per-trial log-likelihoods for one subject."""
            rt = subject_data[:, 0]
            choice = subject_data[:, 1]

            x = jnp.exp(subject_params)
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

            ll = jnp.where(choice == 1, dens_true, dens_false)

            return jnp.sum(jnp.where(subject_mask, ll, 0.0))

        _vmapped_ll = jax.vmap(_single_subject_ll, in_axes=(0, 0, 0))

        @nnx.jit
        def likelihood_fun(x):
            return jnp.sum(_vmapped_ll(data, mask, x))

        return likelihood_fun

    return create_likelihood