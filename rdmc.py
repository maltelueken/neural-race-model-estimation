"""Racing diffusion model for conflict tasks.
"""

import bayesflow as bf
import numpy as np

from bayesflow.utils.decorators import allow_batch_size
from numba import njit, prange
from scipy import stats

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


def rdmc_experiment_simple(v_c_intercept, v_c_slope, amp, tau, s_true, s_false, b, t0, a_shape, num_obs, t_max):
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

    # rt /= 1000

    # timed_out = rt == t_max
    # rt[timed_out] = -1.0
    # resp[timed_out] = -1

    return {"x": np.c_[rt, resp]}


def rdmc_single_accumulator(v_c, amp, tau, s, b, t0, a_shape, num_obs, t_max):
    t = np.arange(1, t_max + 1, 1)

    eq4 = (
        amp
        * np.exp(-t / tau)
        * (np.exp(1) * t / (a_shape - 1) / tau) ** (a_shape - 1)
    ) * ((a_shape - 1) / t - 1 / tau)

    mu = np.tile([v_c], (t_max, 1)).T
    s = np.array([s])

    mu = mu + eq4

    rt, _ = rdmc_experiment_simple_numba(mu, b, s, float(t0), num_obs, t_max)

    return {"x": rt}


def truncated_normal_rvs(
    loc,
    scale,
    lower=0.0,
    size=1,
    random_state=None,
) -> np.ndarray:
    quantile_l = stats.norm.cdf(lower, loc=loc, scale=scale)

    if random_state is not None:
        probs = random_state.uniform(quantile_l, 1.0, size=size)
    else:
        probs = np.random.default_rng().uniform(quantile_l, 1.0, size=size)

    return stats.norm.ppf(
        probs,
        loc=loc,
        scale=scale,
    )


def rdmc_prior(
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


def rdmc_prior_single(
    drift_c_loc=0.5,
    drift_c_scale=0.1,
    amp_shape=10,
    amp_scale=2,
    tau_shape=8,
    tau_scale=10,
    sd_shape=80,
    sd_scale=0.05,
    threshold_shape=100,
    threshold_scale=0.7,
    t0_loc=300,
    t0_scale=200,
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


def random_num_obs(batch_shape, min_obs, max_obs, rng):
    return dict(num_obs=rng.integers(min_obs, max_obs))


class CustomSimulator(bf.simulators.Simulator):
    def __init__(
            self,
            prior_simulator,
            design_simulator,
            experiment_simulator
            ):
        self.prior_simulator = prior_simulator
        self.design_simulator = design_simulator
        self.experiment_simulator = experiment_simulator

    @staticmethod
    def update_dict(d, **kwargs):
        d.update((k, v) for k, v in kwargs.items() if k in d)

    @allow_batch_size
    def sample(self, batch_shape, **kwargs):
        prior_dict = self.prior_simulator.sample(batch_shape)

        self.update_dict(prior_dict, **kwargs)

        design_dict = self.design_simulator.sample(batch_shape)

        self.update_dict(design_dict, **kwargs)

        sims_dict = self.experiment_simulator.sample(batch_shape, **prior_dict, **design_dict)

        data = prior_dict | design_dict | sims_dict

        data = {
            key: np.expand_dims(value, axis=-1) if np.ndim(value) == 1 else value for key, value in data.items()
        }

        return data


def create_rdmc_adapter(param_names):
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

def create_rdmc_adapter_single(param_names):
    return (
        bf.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .drop("num_obs")
        .expand_dims("x", axis=-1)
        .broadcast(param_names, to="x", expand=(1,))
        .as_set(["x"])
        .log(param_names)
        .concatenate(param_names, into="inference_conditions")
        .rename("x", "inference_variables")
    )
