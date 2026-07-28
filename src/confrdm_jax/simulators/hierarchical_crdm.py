import jax
import jax.numpy as jnp

from .base import HierarchicalRDMPriorLKJMVN
from .crdm import simulate_crdm_dataset


NUM_PARAMS = 7


def create_hierarchical_crdm_prior_lkj_mvn(
    num_subjects,
    *,
    inverse_gamma_scale,
    mu_loc,
    mu_scale,
    lkj_concentration=2.0,
):
    """Build the 7-parameter hierarchical CRDM prior.

    The three hyperparameter arrays are required and keyword-only: there is no
    defensible default shared by the RDM (P=5) and CRDM (P=7) layouts, and the
    live values are declared once in ``conf_jax/model/crdm.yaml`` under
    ``hierarchical.prior``. Parameter order is
    ``[v_c_intercept, v_c_slope, amp, tau, s_true, b, t0]``.
    """
    return HierarchicalRDMPriorLKJMVN(
        num_subjects,
        num_params=NUM_PARAMS,
        lkj_concentration=lkj_concentration,
        inverse_gamma_scale=inverse_gamma_scale,
        mu_loc=mu_loc,
        mu_scale=mu_scale,
    )


def sample_conditional_crdm_hierarchical_lkj_mvn(
    key, num_trials, num_subjects,
    *,
    inverse_gamma_scale,
    mu_loc,
    mu_scale,
    lkj_concentration=2.0,
    dt=0.001, t_max=4.0,
):
    """Draw one population from the prior and simulate CRDM data for it.

    Half the trials are congruent (``+amp``) and half incongruent (``-amp``).

    Returns:
        ``(data, context)``. `data` has shape ``(S, num_trials, 3)`` with
        columns ``[rt, choice, condition]``, condition being 1 for congruent
        and 0 for incongruent; `context` is the raw prior sample — a dict with
        keys ``s``, ``mu``, ``psi_raw``, ``z``, ``theta_bt``.
    """
    key_context, key_data_con, key_data_inc = jax.random.split(key, 3)

    prior = create_hierarchical_crdm_prior_lkj_mvn(
        num_subjects,
        inverse_gamma_scale=inverse_gamma_scale,
        mu_loc=mu_loc,
        mu_scale=mu_scale,
        lkj_concentration=lkj_concentration,
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
