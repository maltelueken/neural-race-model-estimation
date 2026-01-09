"""Diffusion model for conflict tasks."""

import numpy as np

from numba import njit, prange
from scipy import stats

from confrdm.utils import truncated_normal_rvs, scaled_gamma_density_derivative

@njit(parallel=True)
def simulate_dmc_numba(mu, b, s, t0, alpha, num_obs, t_max):
    fpt = np.full((num_obs,), fill_value=t_max, dtype=np.float64)
    resp = np.full((num_obs,), fill_value=-1.0, dtype=np.int32)
    start = np.random.beta(alpha, alpha, size=(num_obs,))
    start = -b + start * 2 * b

    for n in prange(num_obs):
        xt = start[n]
        for t in range(t_max):
            xt += mu[t] + s * np.random.randn()
            if xt > b:
                fpt[n] = t
                resp[n] = 1
                break
            elif xt < -b:
                fpt[n] = t
                resp[n] = 0
                break

    rt = fpt + t0

    return rt, resp


def transform_dmc_params(v_c, amp, tau, b, t0, a_shape, s, scale=1000.0):
    scale_sqrt = np.sqrt(scale)

    tau_trans = tau * scale
    t0_trans = t0 * scale
    amp_trans = amp * scale_sqrt
    b_trans = b * scale_sqrt
    v_c_trans = v_c * scale_sqrt / scale

    return v_c_trans, amp_trans, tau_trans, b_trans, t0_trans, a_shape, s


def simulate_dmc(v_c, amp, tau, b, t0, alpha, a_shape, s, num_obs, t_max):
    v_c, amp, tau, b, t0, a_shape, s = transform_dmc_params(v_c, amp, tau, b, t0, a_shape, s)

    t = np.arange(1, t_max + 1, 1)

    v_a = scaled_gamma_density_derivative(t, amp, tau, a_shape)

    mu = v_c + v_a

    rt, resp = simulate_dmc_numba(mu, b, s, float(t0), alpha, num_obs, t_max)

    return {"x": np.c_[rt, resp]}


def sample_dmc_prior(
    drift_loc=3.0,
    drift_scale=0.5,
    amp_loc=0.1,
    amp_scale=0.05,
    tau_loc=0.1,
    tau_scale=0.05,
    threshold_shape=10.0,
    threshold_scale=0.05,
    t0_loc=0.3,
    t0_scale=0.05,
    rng=np.random.default_rng(2025),
):
    drift = truncated_normal_rvs(drift_loc, drift_scale, random_state=rng)
    amp = truncated_normal_rvs(amp_loc, amp_scale, random_state=rng)
    tau = truncated_normal_rvs(tau_loc, tau_scale, random_state=rng)
    b = rng.gamma(shape=threshold_shape, scale=threshold_scale)
    t0 = truncated_normal_rvs(t0_loc, t0_scale, random_state=rng)

    return {
        "v_c": drift,
        "amp": amp,
        "tau": tau,
        "b": b,
        "t0": t0
    }
