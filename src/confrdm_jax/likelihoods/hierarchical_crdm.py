"""Hierarchical conflict diffusion model likelihood.

Combines the condition-dependent neural/analytic routing of
``likelihoods.crdm`` with the ``(data, mask)`` multi-subject interface of
``likelihoods.hierarchical_rdm``; see those two modules for the details of
each.  Parameter order is
``[v_c_intercept, v_c_slope, amp, tau, s_true, b, t0]``, so ``P = 7``.

There is no analytic reference counterpart: the conflict accumulator has a
time-varying drift and no closed-form first-passage density, which is the whole
reason the flow exists.  ``likelihoods.crdm_volterra`` provides a numerical
reference for the single-subject case.

The censoring-sentinel gap documented in ``likelihoods.crdm`` applies here too.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from confrdm_jax.flows import evaluate_pdf_sf
from .rdm import _penalize_invalid_rt
from .rdm import inv_gauss_log_pdf_sf


_FLOOR = 1e-10


def create_crdm_hierarchical_likelihood_factory_approx(conditioner):
    """Factory for hierarchical neural-approximate likelihood for multi-subject CRDM.

    Args:
        conditioner: Trained neural network conditioner for density estimation.

    Returns:
        A function `create_likelihood(data, mask)` returning
        `likelihood_fun(log_theta)`, which takes subject-level log-parameters of
        shape (S, 7) — columns ``[v_c_intercept, v_c_slope, amp, tau, s_true,
        b, t0]`` — and returns a scalar total log-likelihood. It expects `data`
        of shape (S, max_trials, 3) with columns ``[rt, choice, condition]``.
        Population-level parameters do not enter here; the caller reconstructs
        `log_theta` from them first.
    """
    def crdm_log_pdf_sf(rt, v_c, amp, tau, s, b, t0):
        rt_shifted = rt - t0
        rt_safe = jnp.maximum(rt_shifted, _FLOOR)

        v_c = jnp.maximum(v_c, _FLOOR)
        amp = jnp.maximum(amp, _FLOOR)
        tau = jnp.maximum(tau, _FLOOR)
        s = jnp.maximum(s, _FLOOR)
        b = jnp.maximum(b, _FLOOR)

        context = jnp.transpose(jnp.array([v_c, jnp.abs(amp), tau, s, b]))
        log_pdf, log_sf = evaluate_pdf_sf(conditioner, rt_safe, context)
        return _penalize_invalid_rt(rt_shifted, log_pdf, log_sf)

    def create_likelihood(data, mask):
        def _single_subject_ll(subject_data, subject_mask, subject_params):
            """Compute per-trial log-likelihoods for one subject."""
            rt = subject_data[:, 0]
            choice = subject_data[:, 1]
            condition = subject_data[:, 2]  # 1 = Congruent, 0 = Incongruent

            x = jnp.exp(subject_params)
            v_c_true = x[0] + x[1]
            v_c_false = x[0]
            amp = x[2]
            tau = x[3]
            s_true = x[4]
            s_false = 1.0
            b = x[5]
            t0 = x[6]

            # Input routing based on condition
            nn_v_c = jnp.where(condition == 1, v_c_true, v_c_false)
            nn_s = jnp.where(condition == 1, s_true, s_false)
            ig_v_c = jnp.where(condition == 1, v_c_false, v_c_true)
            ig_s = jnp.where(condition == 1, s_false, s_true)

            amp_arr = jnp.tile(amp, nn_v_c.shape)
            tau_arr = jnp.tile(tau, nn_v_c.shape)
            b_arr = jnp.tile(b, nn_v_c.shape)

            # Neural network for the accumulator with conflict modulation
            nn_log_pdf, nn_log_sf = crdm_log_pdf_sf(rt, nn_v_c, amp_arr, tau_arr, nn_s, b_arr, t0)

            # Analytic inverse Gaussian for the other accumulator
            ig_log_pdf, ig_log_sf = inv_gauss_log_pdf_sf(rt, ig_v_c, ig_s, b, t0)

            # Both branches are already clamped and penalised by
            # _penalize_invalid_rt; clamping again here would erase the
            # rt <= t0 penalty gradient.
            nn_log_pdf_c = nn_log_pdf.squeeze()
            nn_log_sf_c = nn_log_sf.squeeze()
            ig_log_pdf_c = ig_log_pdf.squeeze()
            ig_log_sf_c = ig_log_sf.squeeze()

            # Result routing
            log_pdf_true = jnp.where(condition == 1, nn_log_pdf_c, ig_log_pdf_c)
            log_sf_false = jnp.where(condition == 1, ig_log_sf_c, nn_log_sf_c)
            log_pdf_false = jnp.where(condition == 1, ig_log_pdf_c, nn_log_pdf_c)
            log_sf_true = jnp.where(condition == 1, nn_log_sf_c, ig_log_sf_c)

            # Final choice logic
            dens_choice_1 = log_pdf_true + log_sf_false
            dens_choice_0 = log_pdf_false + log_sf_true

            ll = jnp.where(choice == 1, dens_choice_1, dens_choice_0)

            return jnp.sum(jnp.where(subject_mask, ll, 0.0))

        _vmapped_ll = jax.vmap(_single_subject_ll, in_axes=(0, 0, 0))

        @nnx.jit
        def likelihood_fun(x):
            return jnp.sum(_vmapped_ll(data, mask, x))

        return likelihood_fun

    return create_likelihood
