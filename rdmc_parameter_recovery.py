import logging
import os
import time

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"

import blackjax
import bayesflow as bf
import jax
import jax.numpy as jnp
import keras
import numpy as np

import blackjax.smc.resampling as resampling
from blackjax.smc import extend_params

from data import save_hdf5
from mcmc import model_nle
from priors import prior_fun_train, truncated_normal_rvs
from confrdm.simulators.rdmc import simulate_rdmc_two_accumulators
from smc import smc_inference_loop

NUM_TEST_DATASETS = 50
NUM_OBS = 500
NUM_PARTICLES = 1_000
NUM_CHAINS = 2


logger = logging.getLogger()


def simulator_fun(**kwargs):
    return simulate_rdmc_two_accumulators(**kwargs, t_max=2000, seed=2025, a_shape=2, s_false=4)


def meta(batch_size):
    num_obs = np.random.default_rng(2025).integers(250, 500)

    return dict(num_obs=num_obs)


def prior_log_prob(x):
    # x shape: (..., 7)
    v_c_intercept_lp = jax.scipy.stats.norm.logpdf(x[..., 0], loc=0.05, scale=0.05)
    v_c_slope_lp     = jax.scipy.stats.norm.logpdf(x[..., 1], loc=0.5, scale=0.1)
    tau_lp           = jax.scipy.stats.gamma.logpdf(x[..., 2], a=8, scale=10)
    amp_lp           = jax.scipy.stats.gamma.logpdf(x[..., 3], a=10, scale=2)
    s_true_lp        = jax.scipy.stats.gamma.logpdf(x[..., 4], a=80, scale=0.05)
    b_lp             = jax.scipy.stats.gamma.logpdf(x[..., 5], a=100, scale=0.7)
    t0_lp            = jax.scipy.stats.norm.logpdf(x[..., 6], loc=300, scale=50)
    return (
        v_c_intercept_lp
        + v_c_slope_lp
        + tau_lp
        + amp_lp
        + s_true_lp
        + b_lp
        + t0_lp
    )


def sample_prior_particles(n_samples, seed=2025):
    rng = np.random.default_rng(seed)
    v_c_intercept = truncated_normal_rvs(loc=0.05, scale=0.05, size=n_samples, random_state=rng)
    v_c_slope     = truncated_normal_rvs(loc=0.5, scale=0.1, size=n_samples, random_state=rng)
    tau           = rng.gamma(shape=8, scale=10, size=n_samples)
    amp           = rng.gamma(shape=10, scale=2, size=n_samples)
    s_true        = rng.gamma(shape=80, scale=0.05, size=n_samples)
    b             = rng.gamma(shape=100, scale=0.7, size=n_samples)
    t0            = truncated_normal_rvs(loc=300, scale=50, size=n_samples, random_state=rng)
    # Stack in the order of param_names
    return np.stack([v_c_intercept, v_c_slope, tau, amp, s_true, b, t0], axis=-1)


def main():
    param_names = ["v_c_intercept", "v_c_slope", "tau", "amp", "s_true", "b", "t0"]

    simulator = bf.simulators.make_simulator([prior_fun_train, simulator_fun], meta_fn=meta)

    approximator = keras.saving.load_model("checkpoints/rdmc_nle_spline_student_deep_low_lr_50.keras")

    data = []
    final_states = []
    state_histories = []
    true_params = []

    for i in range(NUM_TEST_DATASETS):
        logger.info("Starting dataset %i", i)

        test_data = simulator.sample(1, num_obs=NUM_OBS)
        true = np.array([test_data[p] for p in param_names]).squeeze()

        inv_mass_matrix = jnp.array([
            7.73e-05, 4.44e-04, 2.36e+02, 2.42e+01, 2.25e-01, 2.06e+01, 5.65e+00
        ])

        hmc_parameters = dict(
            step_size=1e-2, inverse_mass_matrix=inv_mass_matrix, num_integration_steps=100
        )

        kernel_fun = blackjax.hmc

        tempered = blackjax.adaptive_tempered_smc(
            prior_log_prob,
            model_nle(test_data, approximator),
            kernel_fun.build_kernel(),
            kernel_fun.init,
            extend_params(hmc_parameters),
            resampling.systematic,
            0.5,
            num_mcmc_steps=1,
        )

        rng_key = jax.random.key(2025+i)

        rng_key, sample_key = jax.random.split(rng_key, 2)
        initial_smc_state = sample_prior_particles(NUM_PARTICLES, seed=2025+i)
        initial_smc_state = tempered.init(initial_smc_state)

        sample_keys = jax.random.split(sample_key, NUM_CHAINS)

        logger.info("Starting inference")
        start_time = time.time()
        n_iter, final_state, state_history = jax.vmap(smc_inference_loop, in_axes=(0, None, None))(sample_keys, tempered.step, initial_smc_state)
        state_history = jax.tree.map(lambda h: h[:, : n_iter[0] + 1], state_history)
        end_time = time.time()
        logger.info("Finished inference in %s", round(end_time - start_time, 2))

        data.append(test_data["x"].squeeze())
        final_states.append(final_state.particles)
        state_histories.append(state_history.particles)
        true_params.append(true)

    results = dict(
        final_state=np.stack(final_states).squeeze(),
        # state_history=state_histories,
        data=np.stack(data).squeeze(),
        true_params=np.stack(true_params).squeeze()
    )
    
    logger.info("Saving results")

    save_hdf5("rdmc_parameter_recovery.hdf5", results)

if __name__ == "__main__":
    main()
