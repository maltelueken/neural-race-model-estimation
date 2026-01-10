"""Racing diffusion model for conflict tasks.
"""

import numpy as np

from numba import njit, prange

from confrdm.utils import truncated_normal_rvs, scaled_gamma_density_derivative

@njit(parallel=True)
def find_min_and_argmin(arr):
    rows, cols = arr.shape
    min_vals = np.empty(cols, dtype=arr.dtype)
    min_idxs = np.empty(cols, dtype=np.int64)
    
    for j in prange(cols):
        min_val = arr[0, j]
        min_idx = 0
        for i in range(1, rows):
            if arr[i, j] < min_val:
                min_val = arr[i, j]
                min_idx = i
        min_vals[j] = min_val
        min_idxs[j] = min_idx
    
    return min_vals, min_idxs


@njit(parallel=True)
def simulate_rdmc_numba(mu, b, s, t0, num_obs, t_max):
    num_accumulators = mu.shape[0]

    fpt = np.full((num_accumulators, num_obs), fill_value=t_max, dtype=np.float64)
    
    for n in prange(num_obs):
        for i in prange(num_accumulators):
            xt = 0.0
            for t in range(t_max):
                xt += mu[i, t] + s[i] * np.random.randn()
                if xt > b:
                    fpt[i, n] = t
                    break

    rt, resp = find_min_and_argmin(fpt)
    rt += t0

    return rt, resp


def transform_rdmc_params(mu_c, amp, tau, s, b, t0, a_shape, scale=1000.0):
    scale_sqrt = np.sqrt(scale)

    tau_trans = tau * scale
    t0_trans = t0 * scale
    amp_trans = amp * scale_sqrt
    b_trans = b * scale_sqrt
    v_c_trans = mu_c * scale_sqrt / scale

    return v_c_trans, amp_trans, tau_trans, s, b_trans, t0_trans, a_shape


def simulate_rdmc_two_accumulators(v_c_intercept, v_c_slope, amp, tau, s_true, s_false, b, t0, a_shape, num_obs, t_max):
    mu_c = np.hstack([v_c_intercept, v_c_intercept + v_c_slope])
    s = np.hstack([s_false, s_true])

    mu_c, amp, tau, s, b, t0, a_shape = transform_rdmc_params(mu_c, amp, tau, s, b, t0, a_shape)

    t = np.arange(1, t_max + 1, 1)

    v_a = scaled_gamma_density_derivative(t, np.abs(amp), tau, a_shape)

    mu = np.tile(mu_c, (t_max, 1)).T

    dim = 1 if amp > 0 else 0

    mu[dim, :] += v_a

    rt, resp = simulate_rdmc_numba(mu, b, s, float(t0), num_obs, t_max)

    return {"x": np.c_[rt, resp]}


def simulate_rdmc_single_accumulator(v_c, amp, tau, s, b, t0, a_shape, num_obs, t_max):
    t = np.arange(1, t_max + 1, 1)

    v_c, amp, tau, s, b, t0, a_shape = transform_rdmc_params(v_c, amp, tau, s, b, t0, a_shape)

    v_a = scaled_gamma_density_derivative(t, amp, tau, a_shape)

    mu = np.tile([v_c], (t_max, 1)).T
    s = np.array([s])[0]

    mu[0, :] + v_a

    rt, _ = simulate_rdmc_numba(mu, b, s, float(t0), num_obs, t_max)

    return {"x": rt}


def sample_rdmc_prior_two_accumulators(
    drift_c_intercept_loc=1.0,
    drift_c_intercept_scale=0.5,
    drift_c_slope_loc=4.0,
    drift_c_slope_scale=0.5,
    amp_loc=0.2,
    amp_scale=0.05,
    tau_loc=0.1,
    tau_scale=0.05,
    sd_true_loc=0.9,
    sd_true_scale=0.1,
    threshold_shape=8.0,
    threshold_scale=0.1,
    t0_loc=0.3,
    t0_scale=0.05,
    rng=np.random.default_rng(2025),
):
    drift_c_intercept = truncated_normal_rvs(drift_c_intercept_loc, drift_c_intercept_scale, random_state=rng)
    drift_c_slope = truncated_normal_rvs(drift_c_slope_loc, drift_c_slope_scale, random_state=rng)
    amp = truncated_normal_rvs(amp_loc, amp_scale, random_state=rng)
    tau = truncated_normal_rvs(tau_loc, tau_scale, random_state=rng)
    s_true = truncated_normal_rvs(sd_true_loc, sd_true_scale, random_state=rng)
    b = rng.gamma(shape=threshold_shape, scale=threshold_scale)
    t0 = truncated_normal_rvs(t0_loc, t0_scale, random_state=rng)

    return {
        "v_c_intercept": drift_c_intercept,
        "v_c_slope": drift_c_slope,
        "amp": amp,
        "tau": tau,
        "s_true": s_true,
        "b": b,
        "t0": t0
    }


def sample_rdmc_prior_single_accumulator(
    drift_c_loc=4.0,
    drift_c_scale=0.5,
    amp_loc=0.2,
    amp_scale=0.05,
    tau_loc=0.1,
    tau_scale=0.05,
    sd_loc=0.9,
    sd_scale=0.1,
    threshold_shape=8.0,
    threshold_scale=0.1,
    t0_loc=0.3,
    t0_scale=0.05,
    rng=np.random.default_rng(2025),
):
    drift_c_slope = truncated_normal_rvs(drift_c_loc, drift_c_scale, random_state=rng)
    amp = truncated_normal_rvs(amp_loc, amp_scale, random_state=rng)
    tau = truncated_normal_rvs(tau_loc, tau_scale, random_state=rng)
    s = truncated_normal_rvs(sd_loc, sd_scale, random_state=rng)
    b = rng.gamma(shape=threshold_shape, scale=threshold_scale)
    t0 = truncated_normal_rvs(t0_loc, t0_scale, random_state=rng)

    return {
        "v_c": drift_c_slope,
        "amp": amp,
        "tau": tau,
        "s": s,
        "b": b,
        "t0": t0
    }
