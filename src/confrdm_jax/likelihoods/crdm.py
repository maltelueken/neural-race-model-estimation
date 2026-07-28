"""Conflict racing diffusion model — hybrid neural / analytic likelihood.

Only one of the two accumulators carries the conflict pulse, and only that one
needs the flow.  The other has constant drift, so its first-passage density is
inverse Gaussian and is computed exactly.  Which accumulator is which flips
with the condition:

=========================  ==========================  ======================
Condition                  Neural flow gets            Inverse Gaussian gets
=========================  ==========================  ======================
congruent (``cond == 1``)  target ``(v_c_true, s_true)``  ``(v_c_false, 1.0)``
incongruent (``cond == 0``)  non-target ``(v_c_false, 1.0)``  ``(v_c_true, s_true)``
=========================  ==========================  ======================

This mirrors the simulator, where the sign of ``amp`` routes the pulse to one
accumulator or the other while its shape always uses ``|amp|``.  The flow
accordingly conditions on ``|amp|`` and the routing is expressed entirely
through which accumulator's ``(v_c, s)`` it is given.

Both branches are evaluated for every trial and selected with ``jnp.where``,
which keeps the computation traceable at the cost of doing roughly twice the
work.  Parameter order is
``[v_c_intercept, v_c_slope, amp, tau, s_true, b, t0]``, in log space on entry.

Known gap:
    The simulators emit ``rt = -1.0`` for trials that never crossed within
    ``t_max``, which is the observation ``T > t_max``. ``flows.loss_fn`` scores
    that correctly by ``log S(t_max)``, but the likelihoods here route it into
    the ``rt <= t0`` branch of ``_penalize_invalid_rt`` instead, contributing a
    large negative penalty with a gradient that drives ``t0`` down. It is
    unreachable under the configured recovery prior — the target accumulator's
    drift of ~5 always crosses long before ``t_max = 4.0`` (measured: 0 of
    10 000 trials censored, longest RT 0.99 s) — but it would become reachable
    if ``t_max`` were shortened or the recovery prior widened toward low drift
    or high boundary.
"""

import jax.numpy as jnp

from confrdm_jax.flows import evaluate_pdf_sf
from .rdm import _penalize_invalid_rt
from .rdm import inv_gauss_log_pdf_sf


_FLOOR = 1e-10


def create_crdm_likelihood_factory_approx(conditioner):
    """Factory that returns a likelihood creator for CRDM using neural approximation.

    Args:
        conditioner: Trained neural network conditioner for density estimation.

    Returns:
        A function that takes data and returns a likelihood function.
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

    def create_likelihood(data):
        rt = data[:, 0]
        choice = data[:, 1]
        condition = data[:, 2]  # 1 = Congruent, 0 = Incongruent

        def likelihood_fun(x):
            x = jnp.exp(x)

            # Define parameters
            v_c_true = x[0] + x[1]
            v_c_false = x[0]
            amp = x[2]
            tau = x[3]
            s_true = x[4]
            s_false = 1.0
            b = x[5]
            t0 = x[6]

            # Input routing based on condition
            # If Congruent: CRDM gets (v_c_true, s_true)
            # If Incongruent: CRDM gets (v_c_false, s_false)
            nn_v_c = jnp.where(condition == 1, v_c_true, v_c_false)
            nn_s = jnp.where(condition == 1, s_true, s_false)

            # Conversely for InvGauss:
            # If Congruent: InvGauss gets (v_c_false, s_false)
            # If Incongruent: InvGauss gets (v_c_true, s_true)
            ig_v_c = jnp.where(condition == 1, v_c_false, v_c_true)
            ig_s = jnp.where(condition == 1, s_false, s_true)

            amp = jnp.tile(amp, nn_v_c.shape)
            tau = jnp.tile(tau, nn_v_c.shape)
            b = jnp.tile(b, nn_v_c.shape)

            # Run Neural Network once per trial with correctly routed params
            nn_log_pdf, nn_log_sf = crdm_log_pdf_sf(rt, nn_v_c, amp, tau, nn_s, b, t0)

            # Run Analytic Function once per trial
            ig_log_pdf, ig_log_sf = inv_gauss_log_pdf_sf(rt, ig_v_c, ig_s, b, t0)

            # Both branches are already clamped and penalised by
            # _penalize_invalid_rt; clamping again here would erase the
            # rt <= t0 penalty gradient.
            nn_log_pdf_c = nn_log_pdf.squeeze()
            nn_log_sf_c = nn_log_sf.squeeze()
            ig_log_pdf_c = ig_log_pdf.squeeze()
            ig_log_sf_c = ig_log_sf.squeeze()

            # Result routing
            # Target Accumulator: If Congruent -> CRDM, If Incongruent -> InvGauss
            log_pdf_true = jnp.where(condition == 1, nn_log_pdf_c, ig_log_pdf_c)

            # Non-Target Accumulator: If Congruent -> InvGauss, If Incongruent -> CRDM
            log_sf_false = jnp.where(condition == 1, ig_log_sf_c, nn_log_sf_c)

            log_pdf_false = jnp.where(condition == 1, ig_log_pdf_c, nn_log_pdf_c)
            log_sf_true = jnp.where(condition == 1, nn_log_sf_c, ig_log_sf_c)

            # Final choice logic
            dens_choice_1 = log_pdf_true + log_sf_false
            dens_choice_0 = log_pdf_false + log_sf_true

            return jnp.where(choice == 1, dens_choice_1, dens_choice_0)

        return likelihood_fun

    return create_likelihood
