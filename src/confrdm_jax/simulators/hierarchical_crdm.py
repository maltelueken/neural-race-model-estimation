import jax
import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd

from .base import HierarchicalRDMPriorLKJMVN, interval_to_mu_loc_scale
from .crdm import simulate_crdm_dataset


def create_hierarchical_crdm_prior_lkj_mvn(
    num_subjects,
    lkj_concentration=2.0,
    inverse_gamma_scale=None,
    mu_loc=None,
    mu_scale=None,
):
    return HierarchicalRDMPriorLKJMVN(
        num_subjects,
        num_params=7,
        lkj_concentration=lkj_concentration,
        inverse_gamma_scale=inverse_gamma_scale,
        mu_loc=mu_loc,
        mu_scale=mu_scale,
    )


def sample_conditional_crdm_hierarchical_lkj_mvn(
    key, num_trials, num_subjects,
    lkj_concentration=2.0,
    inverse_gamma_scale=None,
    mu_loc=None,
    mu_scale=None,
    dt=0.001, t_max=4.0,
):
    key_context, key_data_con, key_data_inc = jax.random.split(key, 3)

    prior = create_hierarchical_crdm_prior_lkj_mvn(
        num_subjects, lkj_concentration=lkj_concentration,
        inverse_gamma_scale=inverse_gamma_scale, mu_loc=mu_loc,
        mu_scale=mu_scale,
    )

    params = prior.sample(seed=key_context)

    log_theta = params["theta"]
    theta = jnp.exp(log_theta)

    context = params

    v_c_intercept = theta[:, 0]
    v_c_slope = theta[:, 1]
    amp = theta[:, 2]
    tau = theta[:, 3]
    s_true = theta[:, 4]
    b = theta[:, 5]
    t0 = theta[:, 6]

    half_trials = num_trials // 2

    # Congruent trials (positive amp)
    keys_con = jax.random.split(key_data_con, num_subjects)

    def _simulate_subject_con(v_int, v_sl, a, ta, s, b_, t0_, k):
        trial_keys = jax.random.split(k, half_trials)
        return simulate_crdm_dataset(
            trial_keys, v_int, v_sl, a, ta, s, b_, t0_, dt, t_max,
        )

    data_con = jax.vmap(_simulate_subject_con)(
        v_c_intercept, v_c_slope, amp, tau, s_true, b, t0, keys_con,
    )

    # Incongruent trials (negative amp)
    keys_inc = jax.random.split(key_data_inc, num_subjects)

    def _simulate_subject_inc(v_int, v_sl, a, ta, s, b_, t0_, k):
        trial_keys = jax.random.split(k, half_trials)
        return simulate_crdm_dataset(
            trial_keys, v_int, v_sl, -a, ta, s, b_, t0_, dt, t_max,
        )

    data_inc = jax.vmap(_simulate_subject_inc)(
        v_c_intercept, v_c_slope, amp, tau, s_true, b, t0, keys_inc,
    )

    # Condition indicator: 1 = congruent, 0 = incongruent
    condition_con = jnp.ones((num_subjects, half_trials, 1))
    condition_inc = jnp.zeros((num_subjects, half_trials, 1))

    # Concatenate: (S, half_trials, 2) -> (S, num_trials, 3)
    data_con_with_cond = jnp.concatenate([data_con, condition_con], axis=-1)
    data_inc_with_cond = jnp.concatenate([data_inc, condition_inc], axis=-1)
    data = jnp.concatenate([data_con_with_cond, data_inc_with_cond], axis=1)

    return data, context
