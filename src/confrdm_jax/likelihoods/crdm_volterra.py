"""CRDM likelihood via Volterra integral equation for first-passage time density.

Solves the Fortet equation numerically to compute the FPT density of a diffusion
process with time-varying drift (gamma-pulse conflict signal) through a constant boundary.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy import stats

from .rdm import inv_gauss_log_pdf_sf, _clamp_log, _penalize_invalid_rt


def scaled_gamma_density(t, amp, tau, a_shape=2.0):
    """Scaled gamma density (antiderivative of gamma_pulse).

    Args:
        t: Time points.
        amp: Amplitude of the conflict signal.
        tau: Time scale (decay rate).
        a_shape: Shape parameter (fixed at 2.0 for CRDM).

    Returns:
        Scaled gamma density evaluated at t.
    """
    return amp * jnp.exp(-t / tau) * (jnp.e * t / ((a_shape - 1) * tau)) ** (a_shape - 1)


def integrated_drift(t, v_c, amp, tau, a_shape=2.0):
    """Integrated drift M(t) = v_c * t + scaled_gamma_density(t).

    Args:
        t: Time points.
        v_c: Constant drift component.
        amp: Amplitude of the conflict signal.
        tau: Time scale.
        a_shape: Shape parameter (fixed at 2.0).

    Returns:
        M(t), the integrated drift at each time point.
    """
    return v_c * t + scaled_gamma_density(t, amp, tau, a_shape)


@partial(jax.jit, static_argnames=["num_steps"])
def solve_volterra_fpt(v_c, amp, tau, s, b, dt, num_steps):
    """Solve the Fortet equation for FPT density on a uniform time grid.

    Solves: h(tₙ) = Σᵢ g(tᵢ) · K(tₙ, tᵢ) · dt
    where h(t) = Φ((M(t) - b) / (σ√t)) and K(t,s) = Φ((M(t) - M(s)) / (σ√(t-s))).

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
    h = stats.norm.cdf((M - b) / (sigma * jnp.sqrt(t_grid)))
    indices = jnp.arange(num_steps)

    # Precompute full lower-triangular kernel matrix — vectorized, outside scan
    t_n = t_grid[:, None]           # (N, 1)
    t_i = t_grid[None, :]           # (1, N)
    dt_diff = jnp.maximum(t_n - t_i, 1e-10)
    M_diff = M[:, None] - M[None, :]
    K_mat = stats.norm.cdf(M_diff / (sigma * jnp.sqrt(dt_diff)))
    K_mat = jnp.tril(K_mat, k=-1)  # zero diagonal and upper triangle

    def scan_fn(g_array, n):
        prev_sum = jnp.dot(g_array, K_mat[n]) * dt

        # Solve for g(t_n): K(t_n, t_n) = 0.5
        g_n = (h[n] - prev_sum) / (dt * 0.5)
        g_n = jnp.maximum(g_n, 0.0)

        g_array = g_array.at[n].set(g_n)
        return g_array, g_n

    g_init = jnp.zeros(num_steps)
    _, g_grid = jax.lax.scan(scan_fn, g_init, indices)

    G_grid = jnp.cumsum(g_grid) * dt

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

    # Steep penalty for impossible observations (rt <= t0)
    # _log_floor = jnp.log(1e-12)
    # valid = rt_shifted > dt
    # penalty = _log_floor + 1e3 * jnp.minimum(rt_shifted - dt, 0.0)
    # log_pdf = jnp.where(valid, log_pdf, penalty)
    # log_sf = jnp.where(valid, log_sf, 0.0)

    return _penalize_invalid_rt(rt_shifted, log_pdf, log_sf)


def create_crdm_likelihood_volterra(data, dt=0.001, t_max=4.0):
    """Create CRDM likelihood using Volterra integral equation solver.

    Follows the same pattern as create_crdm_likelihood_factory_approx: the
    conflicted accumulator uses the Volterra FPT density, the non-conflicted
    accumulator uses the analytical inverse Gaussian.

    Args:
        data: Trial data with columns [RT, choice, condition].
            condition: 1 = Congruent, 0 = Incongruent.
        dt: Time step for Volterra solver.
        t_max: Maximum time for the solver grid.

    Returns:
        A JIT-compiled likelihood function.
    """
    rt = data[:, 0]
    choice = data[:, 1]
    condition = data[:, 2]
    num_steps = int(t_max / dt)

    @jax.jit
    def likelihood_fun(x, min_ll=1e-12):
        x = jnp.exp(x)

        v_c_true = x[0] + x[1]
        v_c_false = x[0]
        amp = x[2]
        tau = x[3]
        s_true = x[4]
        s_false = 1.0
        b = x[5]
        t0 = x[6]

        # Volterra density for congruent condition (true accumulator conflicted)
        con_log_pdf, con_log_sf = crdm_volterra_log_pdf_sf(
            rt, v_c_true, amp, tau, s_true, b, t0, dt, num_steps,
        )

        # Volterra density for incongruent condition (false accumulator conflicted)
        inc_log_pdf, inc_log_sf = crdm_volterra_log_pdf_sf(
            rt, v_c_false, amp, tau, s_false, b, t0, dt, num_steps,
        )

        # Analytical inverse Gaussian for the non-conflicted accumulator
        ig_log_pdf_false, ig_log_sf_false = inv_gauss_log_pdf_sf(rt, v_c_false, s_false, b, t0)
        ig_log_pdf_true, ig_log_sf_true = inv_gauss_log_pdf_sf(rt, v_c_true, s_true, b, t0)

        # Route based on condition
        # Congruent: CRDM=true acc, InvGauss=false acc
        # Incongruent: InvGauss=true acc, CRDM=false acc
        log_pdf_true = jnp.where(condition == 1, con_log_pdf, ig_log_pdf_true)
        log_sf_true = jnp.where(condition == 1, con_log_sf, ig_log_sf_true)
        log_pdf_false = jnp.where(condition == 1, ig_log_pdf_false, inc_log_pdf)
        log_sf_false = jnp.where(condition == 1, ig_log_sf_false, inc_log_sf)

        # Racing likelihood: pdf(winner) * sf(loser)
        dens_choice_1 = _clamp_log(log_pdf_true) + _clamp_log(log_sf_false)
        dens_choice_0 = _clamp_log(log_pdf_false) + _clamp_log(log_sf_true)

        ll = jnp.where(choice == 1, dens_choice_1, dens_choice_0)

        return jnp.where(jnp.isfinite(ll), ll, jnp.log(min_ll))

    return likelihood_fun
