import jax
import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd
from tensorflow_probability.substrates.jax import bijectors as tfb

from .base import HierarchicalRDMPriorLKJMVN, interval_to_mu_loc_scale
from .rdm import simulate_rdm


def create_hierarchical_rdm_prior_lkj_mvn(
    num_subjects,
    lkj_concentration=2.0,
    inverse_gamma_scale=None,
    mu_loc=None,
    mu_scale=None,
):
    return HierarchicalRDMPriorLKJMVN(
        num_subjects,
        num_params=5,
        lkj_concentration=lkj_concentration,
        inverse_gamma_scale=inverse_gamma_scale,
        mu_loc=mu_loc,
        mu_scale=mu_scale,
    )


def sample_conditional_rdm_hierarchical_lkj_mvn(
    key, num_trials,
    num_subjects,
    lkj_concentration=2.0,
    inverse_gamma_scale=None,
    mu_loc=None,
    mu_scale=None,
):
    key_context, key_data = jax.random.split(key)

    prior = create_hierarchical_rdm_prior_lkj_mvn(
        num_subjects, lkj_concentration=lkj_concentration,
        inverse_gamma_scale=inverse_gamma_scale, mu_loc=mu_loc, mu_scale=mu_scale,
    )

    params = prior.sample(seed=key_context)

    # Reconstruct subject-level log-parameters from NCP (z) and centered (theta_bt)
    num_params_ncp = prior.num_params_ncp
    L = params['s'][:, None] * params['psi_raw']           # (P, P)
    L_ncp = L[:num_params_ncp, :num_params_ncp]            # (P_ncp, P_ncp)
    theta_ncp = params['mu'][:num_params_ncp] + jnp.einsum('nj,ij->ni', params['z'], L_ncp)  # (S, P_ncp)
    log_theta = jnp.concatenate([theta_ncp, params['theta_bt']], axis=-1)  # (S, P)
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
