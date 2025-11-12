"""Racing diffusion model for conflict tasks.
"""

from functools import partial

import bayesflow as bf
import distrax
import jax
import jax.numpy as jnp
import keras

from bayesflow.utils.decorators import allow_batch_size

from confrdm.priors import TruncatedNormal 

@jax.jit
def rdmc_single_trial(mu, b, s, t0, key):
    noise = jax.random.normal(key, shape=mu.shape)

    increments = mu + s[:, None] * noise

    x = jnp.cumsum(increments, axis=-1)

    crossing = x > b
    any_cross = jnp.any(crossing, axis=-1)
    first_idx = jnp.argmax(crossing, axis=-1)
    first_time = first_idx + 1
    fpt = jnp.where(any_cross, first_time, mu.shape[1])
    rt = jnp.where(jnp.any(any_cross), jnp.min(fpt, axis=0) + t0, -1.0).astype(jnp.float32)
    resp = jnp.where(jnp.any(any_cross), jnp.argmin(fpt, axis=0), -1).astype(jnp.int32)

    return rt, resp

@partial(jax.jit, static_argnames=["t_max"])
def rdmc_experiment_simple_jax(
        keys,
        v_c_intercept=0.05,
        v_c_slope=0.05,
        amp=20.0,
        tau=80.0,
        s_true=1.5,
        s_false=1.0,
        b=70.0,
        t0=300.0,
        a_shape=2.0,
        t_max=5000,
    ):
    mu_c = jnp.hstack([v_c_intercept, v_c_intercept + v_c_slope])
    s = jnp.hstack([s_false, s_true])

    t = jnp.arange(1, t_max + 1, 1)

    eq4 = (
        amp
        * jnp.exp(-t / tau)
        * (jnp.exp(1) * t / (a_shape - 1) / tau) ** (a_shape - 1)
    ) * ((a_shape - 1) / t - 1 / tau)

    mu = jnp.tile(mu_c, (t_max, 1)).T

    dim = jnp.where(amp > 0.0, 1, 0)

    mu = mu.at[dim, :].add(eq4)

    rt, resp = jax.vmap(rdmc_single_trial, in_axes=(None, None, None, None, 0))(mu, b, s, t0, keys)

    return rt, resp

@partial(jax.jit, static_argnames=["t_max"])
def rdmc_experiment_simple_batched(
        keys,
        v_c_intercept=0.05,
        v_c_slope=0.05,
        amp=20.0,
        tau=80.0,
        s_true=4.0,
        s_false=4.0,
        b=70.0,
        t0=300.0,
        a_shape=2.0,
        t_max=5000,
    ):
    in_axes = [0] * 10 + [None] * 1
    rt, resp = jax.vmap(rdmc_experiment_simple_jax, in_axes=in_axes)(
        keys, v_c_intercept, v_c_slope, amp, tau, s_true, s_false, b, t0, a_shape, t_max
    )

    return rt, resp


class RDMCSimulator(bf.simulators.Simulator):
    param_names = ("v_c_intercept", "v_c_slope", "amp", "tau", "s_true", "b", "t0")

    def __init__(
            self,
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
            t0_loc=300.0,
            t0_scale=200.0,
            s_false=4.0,
            a_shape=2.0,
            num_obs_fun=None,
            t_max=5000,
            start_seed=2025,
        ):
        super().__init__()

        self.prior = distrax.Joint([
            TruncatedNormal(drift_c_intercept_loc, drift_c_intercept_scale, 0.0, jnp.inf), # only lower bound
            TruncatedNormal(drift_c_slope_loc, drift_c_slope_scale, 0.0, jnp.inf),
            distrax.Gamma(amp_shape, 1 / amp_scale),
            distrax.Gamma(tau_shape, 1 / tau_scale),
            distrax.Gamma(sd_true_shape, 1 / sd_true_scale),
            distrax.Gamma(threshold_shape, 1 / threshold_scale),
            TruncatedNormal(t0_loc, t0_scale, 0.0, jnp.inf),
        ])

        self.s_false = s_false
        self.a_shape = a_shape
        self.t_max = t_max

        if num_obs_fun is None:
            raise NotImplementedError("num_obs_fun must not be None")

        self.num_obs_fun = num_obs_fun
        self.simulator_fun = rdmc_experiment_simple_batched
        self.seed_generator = keras.random.SeedGenerator(start_seed)


    @allow_batch_size
    def sample(self, batch_shape, key=None, **kwargs):
        if key is None:
            key = self.seed_generator.next()

        prior_key, num_obs_key, sim_key_master = jax.random.split(key, 3)
        prior_samples = self.prior.sample(seed=prior_key, sample_shape=batch_shape)
        prior_dict = {key: val for key, val in zip(self.param_names, prior_samples)}

        num_obs = self.num_obs_fun(batch_shape, num_obs_key)
        num_obs = kwargs.pop("num_obs", num_obs)
        s_false = kwargs.pop("s_false", self.s_false)
        a_shape = kwargs.pop("a_shape", self.a_shape)

        prior_dict.update(kwargs)

        num_obs_dict = dict(num_obs=num_obs)

        sim_keys = jax.random.split(sim_key_master, batch_shape + (num_obs,))

        joint_dict = prior_dict | num_obs_dict

        rt, resp = self.simulator_fun(
            keys=sim_keys,
            s_false=jnp.full(batch_shape, s_false, dtype=jnp.float32),
            a_shape=jnp.full(batch_shape, a_shape, dtype=jnp.float32),
            t_max=self.t_max,
            **prior_dict
        )

        rt /= 1000.0 # Scale down

        data = dict(x = jnp.stack([rt, resp], axis=2))

        return joint_dict | data
