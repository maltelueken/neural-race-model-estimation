from typing import Tuple

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd
from .rdm import simulate_rdm


def create_rdm_hyperprior_truncated_normal(num_subjects, hyper_mu_mu, hyper_mu_s, hyper_s):
    return  tfd.JointDistributionSequential([
        tfd.TruncatedNormal(hyper_mu_mu, hyper_mu_s, 0.0, jnp.inf),
        tfd.HalfNormal(hyper_s),
        lambda mu, s: tfd.Sample(tfd.TruncatedNormal(mu, s, 0.0, jnp.inf), num_subjects),
    ])


def create_rdm_hyperprior_gamma(num_subjects, hyper_s_mu, hyper_s_s, gamma_shape):
    return  tfd.JointDistributionSequential([
        tfd.TruncatedNormal(hyper_s_mu, hyper_s_s, 0.0, jnp.inf),
        lambda s: tfd.Sample(tfd.Gamma(gamma_shape, 1.0 / s), num_subjects),
    ])


def create_rdm_hierarchical_prior(num_subjects):
    return tfd.JointDistributionSequential([
        create_rdm_hyperprior_truncated_normal(num_subjects, 1.0, 0.25, 0.5),
        create_rdm_hyperprior_truncated_normal(num_subjects, 2.5, 0.25, 0.5),
        create_rdm_hyperprior_gamma(num_subjects, 0.1, 0.05, 12.0),
        create_rdm_hyperprior_gamma(num_subjects, 0.15, 0.05, 8.0),
        create_rdm_hyperprior_truncated_normal(num_subjects, 0.3, 0.2, 0.1),
    ])


def sample_conditional_rdm_hierarchical(
    key: jnp.ndarray,
    num_trials: int,
    prior,
) -> Tuple[jnp.ndarray, list]:
    key_context, key_data = jax.random.split(key)

    context = prior.sample(seed=key_context)

    # Extract per-subject params from the nested sample structure.
    # Truncated normal hyperpriors: (mu, sigma, subject_values)  -> last element
    # Gamma hyperpriors: (scale, subject_values)                 -> last element
    v_intercept = context[0][-1]  # shape (S,)
    v_slope = context[1][-1]      # shape (S,)
    s_true = context[2][-1]       # shape (S,)
    b = context[3][-1]            # shape (S,)
    t0 = context[4][-1]           # shape (S,)

    num_subjects = v_intercept.shape[0]
    keys = jax.random.split(key_data, num_subjects)

    def _simulate_subject(v_int, v_sl, s, b_, t0_, k):
        return simulate_rdm(v_int, v_sl, s, b_, t0_, num_trials, k)

    data = jax.vmap(_simulate_subject)(v_intercept, v_slope, s_true, b, t0, keys)

    return data, context