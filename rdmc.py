"""Racing diffusion model for conflict tasks.
"""

import numpy as np

from numba import njit, prange

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


# @njit
# def set_seed_numba(value):
#     np.random.seed(value)


@njit(parallel=True)
def rdmc_experiment_simple_numba(mu, b, s, t0, num_obs, t_max):
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

    return rt, resp # Convert back to s scale


def rdmc_experiment_simple(v_c_intercept, v_c_slope, amp, tau, s_true, s_false, b, t0, a_shape, num_obs, t_max, seed):
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

    # set_seed_numba(seed)

    rt, resp = rdmc_experiment_simple_numba(mu, b, s, float(t0), num_obs, t_max)

    return {"x": np.c_[rt, resp]}
