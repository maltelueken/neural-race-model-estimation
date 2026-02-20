import jax
import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd
from tensorflow_probability.substrates.jax import bijectors as tfb

from .base import HierarchicalRDMPriorLKJMVN, interval_to_mu_loc_scale
from .rdm import simulate_rdm


def create_hierarchical_rdm_prior_lkj_mvn(
    num_subjects,
    lkj_concentration=2.0,
    halfnormal_scale=None,
    percentile_interval=None,
):
    """Create a hierarchical RDM prior with LKJ-MVN structure.

    Parameters are log-normally distributed. Supply a 95% credible interval
    [P2.5, P97.5] on the original (non-log) scale for each of the 5 RDM
    parameters; the function converts that into Normal hyperparameters
    (mu_loc, mu_scale) in log-space.

    Parameter order: [v_intercept, v_slope, s_true, b, t0]

    Args:
        num_subjects: Number of subjects.
        lkj_concentration: Concentration for CholeskyLKJ prior.
        halfnormal_scale: array-like, shape (5,). Scale of HalfNormal prior on
            between-subject standard deviations in log-space.
        percentile_interval: array-like, shape (5, 2). Each row is [lo, hi] —
            the 2.5th and 97.5th percentile of the marginal prior on that
            parameter (on the original, non-log scale).

    Returns:
        HierarchicalRDMPriorLKJMVN instance.
    """
    mu_loc, mu_scale = interval_to_mu_loc_scale(percentile_interval, halfnormal_scale)
    return HierarchicalRDMPriorLKJMVN(
        num_subjects,
        num_params=5,
        lkj_concentration=lkj_concentration,
        halfnormal_scale=halfnormal_scale,
        mu_loc=mu_loc,
        mu_scale=mu_scale,
    )


def sample_conditional_rdm_hierarchical_lkj_mvn(
    key, num_trials, num_subjects,
    lkj_concentration=2.0, halfnormal_scale=None, percentile_interval=None,
):
    """Sample data from LKJ-MVN hierarchical RDM prior.

    Args:
        key: JAX random key.
        num_trials: Number of trials per subject.
        num_subjects: Number of subjects.
        lkj_concentration: LKJ concentration parameter.
        halfnormal_scale: Scale for HalfNormal prior on std devs (5,).
        percentile_interval: array-like, shape (5, 2). Each row is [lo, hi] —
            the 2.5th and 97.5th percentile of the marginal prior on that
            parameter (on the original, non-log scale).
            Parameter order: [v_intercept, v_slope, s_true, b, t0].

    Returns:
        Tuple of (data, context):
            data: Simulated data array (S, num_trials, 2) with columns [RT, choice].
            context: Dictionary containing hierarchical prior sample.
    """
    key_context, key_data = jax.random.split(key)

    prior = create_hierarchical_rdm_prior_lkj_mvn(
        num_subjects, lkj_concentration=lkj_concentration,
        halfnormal_scale=halfnormal_scale, percentile_interval=percentile_interval,
    )

    params = prior.sample(seed=key_context)

    log_theta = params["theta"]
    theta = jnp.exp(log_theta)

    context = params

    v_intercept = theta[:, 0]
    v_slope = theta[:, 1]
    s_true = theta[:, 2]
    b = theta[:, 3]
    t0 = theta[:, 4]

    keys = jax.random.split(key_data, num_subjects)

    def _simulate_subject(v_int, v_sl, s, b_, t0_, k):
        return simulate_rdm(v_int, v_sl, s, b_, t0_, num_trials, k)

    data = jax.vmap(_simulate_subject)(v_intercept, v_slope, s_true, b, t0, keys)

    return data, context
