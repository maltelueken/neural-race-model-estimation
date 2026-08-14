"""CRDM likelihood via Volterra integral equation for first-passage time density.

Solves the Volterra integral equation of the second kind numerically to compute
the FPT density of a diffusion process with time-varying drift (gamma-pulse conflict
signal) through a constant upper boundary.

This is the **reference** the neural flow is validated against: it computes the
same quantity the flow approximates, by deterministic numerical integration
rather than by learning from simulated trials.  Verified against the analytical
inverse Gaussian in the ``amp -> 0`` limit, where the time-varying drift
degenerates to a constant one: at ``dt = 0.001`` the density agrees to ~2e-11
and the CDF to ~1e-5 across the parameter range used here.

Two ways it is used.  ``scripts/compare_neural_densities.py`` calls
:func:`solve_volterra_fpt` directly to measure flow-vs-reference accuracy on a
parameter grid.  :func:`create_crdm_likelihood_volterra` wraps it into a full
race likelihood with the same neural/analytic routing as
``likelihoods.crdm``; that is a reference posterior path and is **not currently
reachable from Hydra** — ``conf_jax/model/crdm.yaml`` declares no
``likelihood_factory_ref`` and sets ``run_reference_recovery: false``, so
selecting it means adding both.

Being an O(N^2) sequential solve over the time grid, it is far slower than the
flow; that cost is the point of having the flow at all.

Adapted from: integral.py (Richter, Ulrich & Janczyk, 2015).

"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy import stats

from confrdm_jax.simulators.crdm_utils import normalized_gamma, normalized_gamma_derivative
from .rdm import inv_gauss_log_pdf_sf, _penalize_invalid_rt, _censored_eval_rt, _apply_censoring

def integrated_drift(t, v_c, amp, tau, a_shape=2.0):
    """M(t): the accumulator's mean position at time `t`, starting from 0.

    The integral of :func:`instantaneous_drift`. The constant term integrates
    to ``v_c * t``; the pulse integrates back to :func:`normalized_gamma`
    itself, with no constant of integration because ``normalized_gamma(0) = 0``
    for ``a_shape = 2``.
    """
    return v_c * t + normalized_gamma(t, amp, tau, a_shape)


def instantaneous_drift(t, v_c, amp, tau, a_shape=2.0):
    """v(t): the accumulator's drift rate at time `t`.

    Constant baseline plus the conflict pulse — the same drift the
    Euler-Maruyama simulator applies, so the two describe one model.
    """
    return v_c + normalized_gamma_derivative(t, amp, tau, a_shape)


@partial(jax.jit, static_argnames=["num_steps"])
def solve_volterra_fpt(v_c, amp, tau, s, b, dt, num_steps):
    """Solve the Fortet–Smith equation of the second kind for the FPT density.

    Implements the single-boundary, starting-at-zero case of Smith (2000) / Richter
    et al. integral.py.  The equation is:

        g(tₖ) = φ(b, tₖ | 0, 0) · (v(tₖ) + (b − M(tₖ))/tₖ)
                + 2·Δt · Σⱼ₌₁^{k−1} g(tⱼ) · ψ(b, tₖ | b, tⱼ)

    where φ is the Gaussian transition PDF and
        ψ(aᵢ, t | aⱼ, tₐ) = φ(aᵢ, t | aⱼ, tₐ)/2 · (−v(t) + (M(t)−M(tₐ))/(t−tₐ)).

    The ψ kernel vanishes on the diagonal (O(√Δt) → 0), so no special treatment
    of the i=k term is needed and the forward sum runs only over j < k.

    Args:
        v_c: Constant drift component.
        amp: Gamma-pulse amplitude.
        tau: Gamma-pulse time scale.
        s: Diffusion coefficient (noise).
        b: Absorbing boundary.
        dt: Time step size.
        num_steps: Number of time steps.

    Returns:
        Tuple (g_grid, G_grid) where g_grid is the FPT density and G_grid is
        the CDF, both arrays of shape (num_steps,).
    """
    sigma = s
    t_grid = jnp.arange(1, num_steps + 1) * dt
    M = integrated_drift(t_grid, v_c, amp, tau)
    v_inst = instantaneous_drift(t_grid, v_c, amp, tau)

    # Homogeneous (initial) term: φ(b,t|0,0) · (v(t) + (b−M(t))/t)
    sqrt_t = jnp.sqrt(t_grid)
    phi_0 = stats.norm.pdf((b - M) / (sigma * sqrt_t)) / (sigma * sqrt_t)
    h0 = phi_0 * (v_inst + (b - M) / t_grid)

    # Precompute lower-triangular ψ kernel matrix — vectorized, outside scan
    # ψ(b,tₖ|b,tⱼ) = φ(b,tₖ|b,tⱼ)/2 · (−v(tₖ) + (Mₖ−Mⱼ)/(tₖ−tⱼ))
    t_n = t_grid[:, None]           # (N, 1)
    t_j = t_grid[None, :]           # (1, N)
    dt_diff = jnp.maximum(t_n - t_j, 1e-10)
    M_diff = M[:, None] - M[None, :]
    v_n = v_inst[:, None]           # instantaneous drift at tₖ

    phi_mat = stats.norm.pdf(M_diff / (sigma * jnp.sqrt(dt_diff))) / (sigma * jnp.sqrt(dt_diff))
    flux_mat = -v_n + M_diff / dt_diff
    psi_mat = 0.5 * phi_mat * flux_mat

    indices = jnp.arange(num_steps)

    def scan_fn(g_array, n):
        g_n = h0[n] + 2.0 * dt * jnp.dot(g_array, psi_mat[n])
        g_n = jnp.maximum(g_n, 0.0)
        g_array = g_array.at[n].set(g_n)
        return g_array, g_n

    g_init = jnp.zeros(num_steps)
    _, g_grid = jax.lax.scan(scan_fn, g_init, indices)

    # Trapezoidal cumulative integral.  The grid starts at t_1 = dt, and
    # g(0) = 0 exactly for a first-passage density with b > 0, so the missing
    # first panel contributes dt*(g(0) + g_1)/2 and the whole sum reduces to
    # dt*(cumsum(g) - g/2).  The plain right-endpoint rule `cumsum(g)*dt`
    # overestimates G by ~dt*g(t)/2 everywhere — a one-sided bias that shrinks
    # the survival function used for the losing accumulator (measured max
    # G error 0.0072 at dt=0.005, 0.072 for a sharply peaked density).
    G_grid = dt * (jnp.cumsum(g_grid) - 0.5 * g_grid)

    return g_grid, G_grid


@partial(jax.jit, static_argnames=["num_steps"])
def crdm_volterra_log_pdf_sf(rt, v_c, amp, tau, s, b, t0, dt, num_steps):
    """Compute log PDF and log survival function for CRDM via Volterra solver.

    Args:
        rt: Observed response times (array).
        v_c: Drift rate.
        amp: Gamma-pulse amplitude.
        tau: Gamma-pulse time scale.
        s: Diffusion coefficient.
        b: Boundary.
        t0: Non-decision time.
        dt: Time step for Volterra solver.
        num_steps: Number of time steps.

    Returns:
        Tuple (log_pdf, log_sf) evaluated at each rt.
    """
    g_grid, G_grid = solve_volterra_fpt(v_c, amp, tau, s, b, dt, num_steps)
    t_grid = jnp.arange(1, num_steps + 1) * dt

    # Decision time = observed RT minus non-decision time
    rt_shifted = rt - t0
    decision_rt = jnp.maximum(rt_shifted, dt)

    # Interpolate density and CDF at decision times
    g_at_rt = jnp.interp(decision_rt, t_grid, g_grid)
    G_at_rt = jnp.interp(decision_rt, t_grid, G_grid)

    log_pdf = jnp.log(jnp.maximum(g_at_rt, 1e-30))
    log_sf = jnp.log(jnp.maximum(1.0 - G_at_rt, 1e-30))

    return _penalize_invalid_rt(rt_shifted, log_pdf, log_sf)


def create_crdm_likelihood_volterra(data, dt=0.001, t_max=4.0, censor_t_max=None):
    """Create CRDM likelihood using Volterra integral equation solver.

    Follows the same pattern as create_crdm_likelihood_factory_approx: the
    conflicted accumulator uses the Volterra FPT density, the non-conflicted
    accumulator uses the analytical inverse Gaussian.

    Args:
        data: Trial data with columns [RT, choice, condition].
            condition: 1 = Congruent, 0 = Incongruent.
        dt: Time step for Volterra solver.
        t_max: Maximum time for the solver **grid**. This is a numerical
            resolution choice for this solver and is deliberately *not* reused
            as the censoring horizon — the two coincide at their defaults but
            need not, and silently conflating them would misplace the survival
            function if the grid were shortened for speed.
        censor_t_max: Integration horizon of the **simulator** that produced
            `data`, on the decision-time scale. Trials carrying the
            ``rt = -1.0`` sentinel are then scored by
            ``log S_true + log S_false`` at that horizon; see
            :func:`~confrdm_jax.likelihoods.rdm._censored_eval_rt`. ``None``
            leaves the sentinel to the invalid-RT penalty, which is only safe
            for data known to contain none.

    Returns:
        A JIT-compiled likelihood function.
    """
    rt = data[:, 0]
    choice = data[:, 1]
    condition = data[:, 2]
    num_steps = int(t_max / dt)

    @jax.jit
    def likelihood_fun(x):
        x = jnp.exp(x)

        v_c_true = x[0] + x[1]
        v_c_false = x[0]
        amp = x[2]
        tau = x[3]
        s_true = x[4]
        s_false = 1.0
        b = x[5]
        t0 = x[6]

        # Censored trials are evaluated at the simulator's horizon instead of
        # at the sentinel, so every accumulator yields S(censor_t_max).
        rt_eval, is_censored = _censored_eval_rt(rt, t0, censor_t_max)

        # Volterra density for congruent condition (true accumulator conflicted)
        con_log_pdf, con_log_sf = crdm_volterra_log_pdf_sf(
            rt_eval, v_c_true, amp, tau, s_true, b, t0, dt, num_steps,
        )

        # Volterra density for incongruent condition (false accumulator conflicted)
        inc_log_pdf, inc_log_sf = crdm_volterra_log_pdf_sf(
            rt_eval, v_c_false, amp, tau, s_false, b, t0, dt, num_steps,
        )

        # Analytical inverse Gaussian for the non-conflicted accumulator
        ig_log_pdf_false, ig_log_sf_false = inv_gauss_log_pdf_sf(
            rt_eval, v_c_false, s_false, b, t0,
        )
        ig_log_pdf_true, ig_log_sf_true = inv_gauss_log_pdf_sf(
            rt_eval, v_c_true, s_true, b, t0,
        )

        # Route based on condition
        # Congruent: CRDM=true acc, InvGauss=false acc
        # Incongruent: InvGauss=true acc, CRDM=false acc
        log_pdf_true = jnp.where(condition == 1, con_log_pdf, ig_log_pdf_true)
        log_sf_true = jnp.where(condition == 1, con_log_sf, ig_log_sf_true)
        log_pdf_false = jnp.where(condition == 1, ig_log_pdf_false, inc_log_pdf)
        log_sf_false = jnp.where(condition == 1, ig_log_sf_false, inc_log_sf)

        # Racing likelihood: pdf(winner) * sf(loser).
        # Already clamped/penalised upstream; see _penalize_invalid_rt.
        dens_choice_1 = log_pdf_true + log_sf_false
        dens_choice_0 = log_pdf_false + log_sf_true

        ll = jnp.where(choice == 1, dens_choice_1, dens_choice_0)

        # A censored trial has no winner: both accumulators survive to the
        # horizon, so it contributes both survival factors and no density.
        return _apply_censoring(is_censored, log_sf_true, log_sf_false, ll)

    return likelihood_fun
