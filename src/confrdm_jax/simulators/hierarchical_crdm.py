import jax
import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd

from .base import HierarchicalRDMPriorLKJMVN, interval_to_mu_loc_scale
from .crdm import simulate_crdm_dataset


def create_hierarchical_crdm_prior_lkj_mvn(
    num_subjects,
    lkj_concentration=2.0,
    halfnormal_scale=None,
    percentile_interval=None,
):
    """Create a hierarchical CRDM prior with LKJ-MVN structure.

    Parameters are log-normally distributed. Supply a 95% credible interval
    [P2.5, P97.5] on the original (non-log) scale for each of the 7 CRDM
    parameters; the function converts that into Normal hyperparameters
    (mu_loc, mu_scale) in log-space.

    Parameter order: [v_c_intercept, v_c_slope, amp, tau, s_true, b, t0]

    Args:
        num_subjects: Number of subjects.
        lkj_concentration: Concentration for CholeskyLKJ prior.
        halfnormal_scale: array-like, shape (7,). Scale of HalfNormal prior on
            between-subject standard deviations in log-space.
        percentile_interval: array-like, shape (7, 2). Each row is [lo, hi] —
            the 2.5th and 97.5th percentile of the marginal prior on that
            parameter (on the original, non-log scale).

    Returns:
        HierarchicalRDMPriorLKJMVN instance.
    """
    mu_loc, mu_scale = interval_to_mu_loc_scale(percentile_interval, halfnormal_scale)
    return HierarchicalRDMPriorLKJMVN(
        num_subjects,
        num_params=7,
        lkj_concentration=lkj_concentration,
        halfnormal_scale=halfnormal_scale,
        mu_loc=mu_loc,
        mu_scale=mu_scale,
    )


def sample_conditional_crdm_hierarchical_lkj_mvn(
    key, num_trials, num_subjects,
    lkj_concentration=2.0, halfnormal_scale=None, percentile_interval=None,
    dt=0.001, t_max=4.0,
):
    """Sample data from LKJ-MVN hierarchical CRDM prior.

    Simulates half congruent and half incongruent trials per subject,
    concatenated with a condition indicator column.

    Args:
        key: JAX random key.
        num_trials: Total number of trials per subject (half congruent, half incongruent).
        num_subjects: Number of subjects.
        lkj_concentration: LKJ concentration parameter.
        halfnormal_scale: Scale for HalfNormal prior on std devs (7,).
        percentile_interval: array-like, shape (7, 2). Each row is [lo, hi] —
            the 2.5th and 97.5th percentile of the marginal prior on that
            parameter (on the original, non-log scale).
            Parameter order: [v_c_intercept, v_c_slope, amp, tau, s_true, b, t0].
        dt: Time step for simulation.
        t_max: Maximum simulation time.

    Returns:
        Tuple of (data, context):
            data: Simulated data array (S, num_trials, 3) with columns [RT, choice, condition].
            context: Dictionary containing hierarchical prior sample.
    """
    key_context, key_data_con, key_data_inc = jax.random.split(key, 3)

    prior = create_hierarchical_crdm_prior_lkj_mvn(
        num_subjects, lkj_concentration=lkj_concentration,
        halfnormal_scale=halfnormal_scale, percentile_interval=percentile_interval,
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
