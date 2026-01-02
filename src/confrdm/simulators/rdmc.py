"""Racing diffusion model for conflict tasks.
"""

import bayesflow as bf
import numpy as np

from numba import njit, prange

from confrdm.utils import truncated_normal_rvs

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


def simulate_rdmc_two_accumulators(v_c_intercept, v_c_slope, amp, tau, s_true, s_false, b, t0, a_shape, num_obs, t_max):
    mu_c = np.hstack([v_c_intercept, v_c_intercept + v_c_slope])
    s = np.hstack([s_false, s_true])

    t = np.arange(1, t_max + 1, 1)

    eq4 = (
        amp
        * np.exp(-t / tau)
        * (np.exp(1) * t / (a_shape - 1) / tau) ** (a_shape - 1)
    ) * ((a_shape - 1) / t - 1 / tau)

    mu = np.tile(mu_c, (t_max, 1)).T

    dim = 1 if amp > 0 else 0

    mu[dim, :] += eq4

    rt, resp = simulate_rdmc_numba(mu, b, s, float(t0), num_obs, t_max)

    return {"x": np.c_[rt, resp]}


def simulate_rdmc_single_accumulator(v_c, amp, tau, s, b, t0, a_shape, num_obs, t_max):
    t = np.arange(1, t_max + 1, 1)

    eq4 = (
        amp
        * np.exp(-t / tau)
        * (np.exp(1) * t / (a_shape - 1) / tau) ** (a_shape - 1)
    ) * ((a_shape - 1) / t - 1 / tau)

    mu = np.tile([v_c], (t_max, 1)).T
    s = np.array([s])

    mu = mu + eq4

    rt, _ = simulate_rdmc_numba(mu, b, s, float(t0), num_obs, t_max)

    return {"x": rt}


def sample_rdmc_prior_two_accumulators(
    drift_c_intercept_loc=0.05,
    drift_c_intercept_scale=0.05,
    drift_c_slope_loc=0.5,
    drift_c_slope_scale=0.1,
    amp_shape=10,
    amp_scale=2,
    tau_shape=8,
    tau_scale=10,
    sd_true_shape=80,
    sd_true_scale=0.05,
    threshold_shape=100,
    threshold_scale=0.7,
    t0_loc=300,
    t0_scale=200,
    rng=np.random.default_rng(2025),
):
    drift_c_intercept = truncated_normal_rvs(drift_c_intercept_loc, drift_c_intercept_scale, random_state=rng)
    drift_c_slope = truncated_normal_rvs(drift_c_slope_loc, drift_c_slope_scale, random_state=rng)
    amp = rng.gamma(shape=amp_shape, scale=amp_scale)
    tau = rng.gamma(shape=tau_shape, scale=tau_scale)
    s_true = rng.gamma(shape=sd_true_shape, scale=sd_true_scale)
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
    drift_c_loc=0.5,
    drift_c_scale=1.0,
    amp_shape=2,
    amp_scale=10,
    tau_shape=0.8,
    tau_scale=100,
    sd_shape=4,
    sd_scale=1,
    threshold_shape=10,
    threshold_scale=7,
    t0_loc=300,
    t0_scale=500,
    rng=np.random.default_rng(2025),
):
    drift_c_slope = truncated_normal_rvs(drift_c_loc, drift_c_scale, random_state=rng)
    amp = rng.gamma(shape=amp_shape, scale=amp_scale)
    tau = rng.gamma(shape=tau_shape, scale=tau_scale)
    s = rng.gamma(shape=sd_shape, scale=sd_scale)
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


def create_nle_adapter_two_accumulators(param_names):
    return (
        bf.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .broadcast(param_names, to="x", expand=1)
        .drop("num_obs")
        .as_set(["x"])
        .log(param_names)
        .concatenate(param_names, into="inference_conditions")
        .rename("x", "inference_variables")
    )


def create_nle_adapter_single_accumulator(param_names):
    return (
        bf.Adapter()
        .to_array()
        # .convert_dtype("float64", "float32")
        .drop("num_obs")
        .expand_dims("x", axis=-1)
        .broadcast(param_names, to="x", expand=(1,))
        .as_set(["x"])
        .log(param_names)
        .concatenate(param_names, into="inference_conditions")
        .rename("x", "inference_variables")
    )


def create_npe_adapter_two_accumulators(param_names):
    return (
        bf.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .broadcast("num_obs", to="x", exclude=(-2, -1), squeeze=-1)
        .as_set(["x"])
        .log(param_names)
        .sqrt("num_obs")
        .concatenate(param_names, into="inference_variables")
        .concatenate(["x"], into="summary_variables")
        .rename("num_obs", "inference_conditions")
        .keep(
            ["inference_variables", "inference_conditions", "summary_variables"]
        )
    )
