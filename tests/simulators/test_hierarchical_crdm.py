"""Tests for hierarchical CRDM simulator and LKJ-MVN prior.

The prior class is shared with the RDM (``HierarchicalRDMPriorLKJMVN``); what is
specific here is the P=7 layout and the congruent/incongruent split in the
simulated data. The simulator is Euler-Maruyama, so these tests use a coarse
``dt`` and a short ``t_max`` to stay fast.
"""

import jax.numpy as jnp
import pytest

from confrdm_jax.simulators import (
    create_hierarchical_crdm_prior_lkj_mvn,
    sample_conditional_crdm_hierarchical_lkj_mvn,
)


NUM_PARAMS = 7
NUM_CENTERED = 2
NUM_PARAMS_NCP = NUM_PARAMS - NUM_CENTERED

# Mirrors ``hierarchical.prior`` in conf_jax/model/crdm.yaml.
# Parameter order: [v_c_intercept, v_c_slope, amp, tau, s_true, b, t0].
PRIOR_KWARGS = {
    "lkj_concentration": 2.0,
    "inverse_gamma_scale": jnp.array([0.4, 0.8, 0.8, 0.4, 0.4, 0.4, 0.25]),
    "mu_loc": jnp.array([0.1, 0.5, -1.4, -2.2, 0.0, 0.0, -1.2]),
    "mu_scale": jnp.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.1]),
}

SAMPLE_KEYS = {"s", "mu", "psi_raw", "z", "theta_bt"}

SIM_KWARGS = {"dt": 0.01, "t_max": 2.0}


def reconstruct_log_theta(params):
    """Rebuild subject log-parameters from a prior sample."""
    L = params["s"][:, None] * params["psi_raw"]
    L_ncp = L[:NUM_PARAMS_NCP, :NUM_PARAMS_NCP]
    theta_ncp = params["mu"][:NUM_PARAMS_NCP] + jnp.einsum(
        "nj,ij->ni", params["z"], L_ncp,
    )
    return jnp.concatenate([theta_ncp, params["theta_bt"]], axis=-1)


class TestCreateHierarchicalCrdmPriorLkjMvn:

    def test_sample_shapes(self, rng_key):
        num_subjects = 3
        prior = create_hierarchical_crdm_prior_lkj_mvn(num_subjects, **PRIOR_KWARGS)
        sample = prior.sample(seed=rng_key)

        assert set(sample.keys()) == SAMPLE_KEYS
        assert sample["s"].shape == (NUM_PARAMS,)
        assert sample["mu"].shape == (NUM_PARAMS,)
        assert sample["psi_raw"].shape == (NUM_PARAMS, NUM_PARAMS)
        assert sample["z"].shape == (num_subjects, NUM_PARAMS_NCP)
        assert sample["theta_bt"].shape == (num_subjects, NUM_CENTERED)

    def test_log_prob_is_finite(self, rng_key):
        prior = create_hierarchical_crdm_prior_lkj_mvn(3, **PRIOR_KWARGS)
        assert jnp.isfinite(prior.log_prob(prior.sample(seed=rng_key)))

    def test_hyperparameters_are_required(self):
        with pytest.raises(TypeError):
            create_hierarchical_crdm_prior_lkj_mvn(num_subjects=3)

    def test_rdm_hyperparameters_are_rejected(self):
        """A 5-parameter layout must not silently be accepted for P=7."""
        bad = {k: v[:5] if hasattr(v, "shape") else v for k, v in PRIOR_KWARGS.items()}
        with pytest.raises(ValueError, match="num_params"):
            create_hierarchical_crdm_prior_lkj_mvn(3, **bad)


class TestSampleConditionalCrdmHierarchicalLkjMvn:

    def test_data_shape_and_condition_split(self, rng_key):
        """Columns are [rt, choice, condition], half congruent / half incongruent."""
        num_subjects, num_trials = 2, 20
        data, _ = sample_conditional_crdm_hierarchical_lkj_mvn(
            rng_key, num_trials, num_subjects, **PRIOR_KWARGS, **SIM_KWARGS,
        )

        assert data.shape == (num_subjects, num_trials, 3)

        condition = data[..., 2]
        assert jnp.all((condition == 0) | (condition == 1))
        assert jnp.all(jnp.sum(condition, axis=-1) == num_trials // 2)

    def test_choices_are_binary(self, rng_key):
        data, _ = sample_conditional_crdm_hierarchical_lkj_mvn(
            rng_key, 20, 2, **PRIOR_KWARGS, **SIM_KWARGS,
        )
        choices = data[..., 1]
        assert jnp.all((choices == 0) | (choices == 1))

    def test_rt_is_sentinel_or_above_t0(self, rng_key):
        """Trials that never cross carry the -1 sentinel; the rest must exceed t0."""
        num_subjects = 2
        data, context = sample_conditional_crdm_hierarchical_lkj_mvn(
            rng_key, 20, num_subjects, **PRIOR_KWARGS, **SIM_KWARGS,
        )
        t0 = jnp.exp(reconstruct_log_theta(context))[:, 6]

        rt = data[..., 0]
        crossed = rt > 0
        assert jnp.all(jnp.where(crossed, rt, jnp.inf) > t0[:, None])
        assert jnp.all(jnp.where(crossed, -1.0, rt) == -1.0)

    def test_context_has_valid_log_prob(self, rng_key):
        num_subjects = 2
        prior = create_hierarchical_crdm_prior_lkj_mvn(num_subjects, **PRIOR_KWARGS)
        _, context = sample_conditional_crdm_hierarchical_lkj_mvn(
            rng_key, 20, num_subjects, **PRIOR_KWARGS, **SIM_KWARGS,
        )
        assert set(context.keys()) == SAMPLE_KEYS
        assert reconstruct_log_theta(context).shape == (num_subjects, NUM_PARAMS)
        assert jnp.isfinite(prior.log_prob(context))

    def test_reproducibility(self, rng_key):
        data1, ctx1 = sample_conditional_crdm_hierarchical_lkj_mvn(
            rng_key, 20, 2, **PRIOR_KWARGS, **SIM_KWARGS,
        )
        data2, ctx2 = sample_conditional_crdm_hierarchical_lkj_mvn(
            rng_key, 20, 2, **PRIOR_KWARGS, **SIM_KWARGS,
        )
        assert jnp.allclose(data1, data2)
        for k in ctx1:
            assert jnp.allclose(ctx1[k], ctx2[k])
