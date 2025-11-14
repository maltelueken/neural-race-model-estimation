"""Functions for MCMC estimation.
"""

from datetime import date

import blackjax
import keras
import jax
import jax.numpy as jnp
import numpy as np

def model_nle(data, approximator, fixed=None):
    num_obs = data["num_obs"]
    adapted = approximator.adapter(data, strict=False, log_det_jac=True, stage="inference")
    data, ldj = adapted
    log_det_jac = ldj.get("inference_variables", 0.0)

    @jax.jit
    def nle_logdensity_fun(x):
        sim_data = data.copy()

        if fixed is not None:
            for p in fixed:
                x = x.at[p[0]].set(p[1])

        sim_data["inference_conditions"] = jnp.tile(x, (1, num_obs, 1))

        for key in approximator.CONDITION_KEYS:
            if key in approximator.standardize and key in sim_data:
                sim_data[key] = approximator.standardize_layers[key](sim_data[key])

        # ldj = 0.0

        if "inference_variables" in sim_data and "inference_variables" in approximator.standardize:
            result = approximator.standardize_layers["inference_variables"](
                sim_data["inference_variables"], log_det_jac=True
            )
            sim_data["inference_variables"], ldj = result

        log_prob = approximator._log_prob(**sim_data)

        log_prob = log_prob + ldj

        return log_prob.sum()

    return nle_logdensity_fun


def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)

    return states, infos


def inference_loop_multiple_chains(
    rng_key, kernel, initial_state, num_samples, num_chains
):

    @jax.jit
    def one_step(states, rng_key):
        keys = jax.random.split(rng_key, num_chains)
        states, _ = jax.vmap(kernel)(keys, states)
        return states, states

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


def warmup(sampler_fun, logdensity_fun, init_position, num_steps, rng_key, **kwargs):
    adapt = blackjax.window_adaptation(sampler_fun, logdensity_fun, **kwargs)
    (last_state, parameters), _ = adapt.run(rng_key, init_position, num_steps=num_steps)
    kernel = sampler_fun(logdensity_fun, **parameters).step
    return kernel, last_state, parameters
