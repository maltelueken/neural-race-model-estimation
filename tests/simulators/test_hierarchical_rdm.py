"""Tests for hierarchical RDM simulator and LKJ-MVN prior."""

import jax
import jax.numpy as jnp
import pytest

from confrdm_jax.simulators import (
    create_hierarchical_rdm_prior_lkj_mvn,
    sample_conditional_rdm_hierarchical_lkj_mvn,
)


NUM_PARAMS = 5
NUM_CENTERED = 2
NUM_PARAMS_NCP = NUM_PARAMS - NUM_CENTERED

# Mirrors ``hierarchical.prior`` in conf_jax/model/rdm.yaml, so the tests
# exercise the hyperparameters the pipeline actually runs with.
# Parameter order: [v_intercept, v_slope, s_true, b, t0].
PRIOR_KWARGS = {
    "lkj_concentration": 2.0,
    "inverse_gamma_scale": jnp.array([0.4, 0.4, 0.4, 0.4, 0.25]),
    "mu_loc": jnp.array([0.1, 0.3, 0.4, 0.4, -1.2]),
    "mu_scale": jnp.array([0.25, 0.25, 0.25, 0.25, 0.1]),
}

SAMPLE_KEYS = {"s", "mu", "psi_raw", "z", "theta_bt"}


def reconstruct_log_theta(params):
    """Rebuild subject log-parameters from a prior sample.

    Same reconstruction the simulator and the SMC script perform: the leading
    block is non-centered (``z``), the trailing block centered (``theta_bt``).
    """
    L = params["s"][:, None] * params["psi_raw"]
    L_ncp = L[:NUM_PARAMS_NCP, :NUM_PARAMS_NCP]
    theta_ncp = params["mu"][:NUM_PARAMS_NCP] + jnp.einsum(
        "nj,ij->ni", params["z"], L_ncp,
    )
    return jnp.concatenate([theta_ncp, params["theta_bt"]], axis=-1)


class TestCreateHierarchicalRdmPriorLkjMvn:
    """Tests for create_hierarchical_rdm_prior_lkj_mvn."""

    def test_sample_returns_named_components(self, rng_key):
        prior = create_hierarchical_rdm_prior_lkj_mvn(num_subjects=3, **PRIOR_KWARGS)
        sample = prior.sample(seed=rng_key)
        assert set(sample.keys()) == SAMPLE_KEYS

    def test_sample_shapes(self, rng_key):
        num_subjects = 4
        prior = create_hierarchical_rdm_prior_lkj_mvn(
            num_subjects=num_subjects, **PRIOR_KWARGS,
        )
        sample = prior.sample(seed=rng_key)

        assert sample["s"].shape == (NUM_PARAMS,)
        assert sample["mu"].shape == (NUM_PARAMS,)
        assert sample["psi_raw"].shape == (NUM_PARAMS, NUM_PARAMS)
        assert sample["z"].shape == (num_subjects, NUM_PARAMS_NCP)
        assert sample["theta_bt"].shape == (num_subjects, NUM_CENTERED)

    def test_s_positive(self, rng_key):
        """Between-subject scales are InverseGamma, hence strictly positive."""
        prior = create_hierarchical_rdm_prior_lkj_mvn(num_subjects=3, **PRIOR_KWARGS)
        assert jnp.all(prior.sample(seed=rng_key)["s"] > 0)

    def test_psi_raw_is_a_correlation_cholesky(self, rng_key):
        """CholeskyLKJ draws are lower triangular with unit-norm rows."""
        prior = create_hierarchical_rdm_prior_lkj_mvn(num_subjects=3, **PRIOR_KWARGS)
        psi_raw = prior.sample(seed=rng_key)["psi_raw"]

        assert jnp.allclose(psi_raw, jnp.tril(psi_raw))
        assert jnp.allclose(jnp.sum(psi_raw ** 2, axis=-1), 1.0, atol=1e-5)

    def test_log_prob_is_finite(self, rng_key):
        prior = create_hierarchical_rdm_prior_lkj_mvn(num_subjects=3, **PRIOR_KWARGS)
        lp = prior.log_prob(prior.sample(seed=rng_key))
        assert jnp.isfinite(lp)

    def test_num_subjects_changes_subject_block_shapes(self, rng_key):
        prior_2 = create_hierarchical_rdm_prior_lkj_mvn(num_subjects=2, **PRIOR_KWARGS)
        prior_7 = create_hierarchical_rdm_prior_lkj_mvn(num_subjects=7, **PRIOR_KWARGS)

        log_theta_2 = reconstruct_log_theta(prior_2.sample(seed=rng_key))
        log_theta_7 = reconstruct_log_theta(prior_7.sample(seed=rng_key))

        assert log_theta_2.shape == (2, NUM_PARAMS)
        assert log_theta_7.shape == (7, NUM_PARAMS)

    def test_mode_shapes_and_values(self):
        """mode() is what MCMC/SMC initialisation ravels; check it analytically."""
        num_subjects = 3
        prior = create_hierarchical_rdm_prior_lkj_mvn(
            num_subjects=num_subjects, **PRIOR_KWARGS,
        )
        mode = prior.mode()

        assert set(mode.keys()) == SAMPLE_KEYS
        # InverseGamma(alpha=4, beta): mode = beta / (alpha + 1) = beta / 5
        assert jnp.allclose(mode["s"], PRIOR_KWARGS["inverse_gamma_scale"] / 5.0)
        assert jnp.allclose(mode["mu"], PRIOR_KWARGS["mu_loc"])
        assert jnp.allclose(mode["psi_raw"], jnp.eye(NUM_PARAMS))
        assert jnp.allclose(mode["z"], jnp.zeros((num_subjects, NUM_PARAMS_NCP)))
        assert jnp.allclose(
            mode["theta_bt"],
            jnp.broadcast_to(
                PRIOR_KWARGS["mu_loc"][NUM_PARAMS_NCP:],
                (num_subjects, NUM_CENTERED),
            ),
        )

    def test_hyperparameters_are_required(self):
        """No silent defaults: omitting them must fail loudly at the call site."""
        with pytest.raises(TypeError):
            create_hierarchical_rdm_prior_lkj_mvn(num_subjects=3)

    def test_hyperparameter_length_is_validated(self):
        bad = dict(PRIOR_KWARGS, mu_loc=jnp.zeros(NUM_PARAMS + 1))
        with pytest.raises(ValueError, match="num_params"):
            create_hierarchical_rdm_prior_lkj_mvn(num_subjects=3, **bad)


class TestSampleConditionalRdmHierarchicalLkjMvn:
    """Tests for sample_conditional_rdm_hierarchical_lkj_mvn."""

    def test_data_shape(self, rng_key):
        num_subjects, num_trials = 3, 100
        data, _ = sample_conditional_rdm_hierarchical_lkj_mvn(
            rng_key, num_trials, num_subjects, **PRIOR_KWARGS,
        )
        assert data.shape == (num_subjects, num_trials, 2)

    def test_data_is_finite(self, rng_key):
        data, _ = sample_conditional_rdm_hierarchical_lkj_mvn(
            rng_key, 50, num_subjects=3, **PRIOR_KWARGS,
        )
        assert jnp.all(jnp.isfinite(data))

    def test_rt_greater_than_t0(self, rng_key):
        """All RTs should exceed the simulating subject's own t0."""
        num_subjects = 3
        data, context = sample_conditional_rdm_hierarchical_lkj_mvn(
            rng_key, 200, num_subjects, **PRIOR_KWARGS,
        )
        t0_per_subject = jnp.exp(reconstruct_log_theta(context))[:, 4]
        for s in range(num_subjects):
            assert jnp.all(data[s, :, 0] > t0_per_subject[s])

    def test_choices_are_binary(self, rng_key):
        data, _ = sample_conditional_rdm_hierarchical_lkj_mvn(
            rng_key, 100, num_subjects=3, **PRIOR_KWARGS,
        )
        choices = data[:, :, 1]
        assert jnp.all((choices == 0) | (choices == 1))

    def test_context_keys(self, rng_key):
        """Context is the raw prior sample, not a derived summary."""
        _, context = sample_conditional_rdm_hierarchical_lkj_mvn(
            rng_key, 50, 4, **PRIOR_KWARGS,
        )
        assert set(context.keys()) == SAMPLE_KEYS

    def test_context_shapes(self, rng_key):
        num_subjects = 4
        _, context = sample_conditional_rdm_hierarchical_lkj_mvn(
            rng_key, 50, num_subjects, **PRIOR_KWARGS,
        )
        assert context["s"].shape == (NUM_PARAMS,)
        assert context["mu"].shape == (NUM_PARAMS,)
        assert context["psi_raw"].shape == (NUM_PARAMS, NUM_PARAMS)
        assert context["z"].shape == (num_subjects, NUM_PARAMS_NCP)
        assert context["theta_bt"].shape == (num_subjects, NUM_CENTERED)
        assert reconstruct_log_theta(context).shape == (num_subjects, NUM_PARAMS)

    def test_context_has_valid_log_prob(self, rng_key):
        """The returned context should be a valid sample under the same prior."""
        num_subjects = 3
        prior = create_hierarchical_rdm_prior_lkj_mvn(num_subjects, **PRIOR_KWARGS)
        _, context = sample_conditional_rdm_hierarchical_lkj_mvn(
            rng_key, 50, num_subjects, **PRIOR_KWARGS,
        )
        assert jnp.isfinite(prior.log_prob(context))

    def test_reproducibility(self, rng_key):
        """Same key should produce identical data and context."""
        num_subjects = 3
        data1, ctx1 = sample_conditional_rdm_hierarchical_lkj_mvn(
            rng_key, 50, num_subjects, **PRIOR_KWARGS,
        )
        data2, ctx2 = sample_conditional_rdm_hierarchical_lkj_mvn(
            rng_key, 50, num_subjects, **PRIOR_KWARGS,
        )
        assert jnp.allclose(data1, data2)
        for k in ctx1:
            assert jnp.allclose(ctx1[k], ctx2[k])

    def test_different_keys_produce_different_data(self, rng_key):
        key2 = jax.random.split(rng_key)[0]

        data1, _ = sample_conditional_rdm_hierarchical_lkj_mvn(
            rng_key, 100, 3, **PRIOR_KWARGS,
        )
        data2, _ = sample_conditional_rdm_hierarchical_lkj_mvn(
            key2, 100, 3, **PRIOR_KWARGS,
        )
        assert not jnp.allclose(data1, data2)

    def test_different_num_subjects(self, rng_key):
        """Different subject counts should produce correctly shaped data."""
        for n in [1, 2, 5, 10]:
            data, context = sample_conditional_rdm_hierarchical_lkj_mvn(
                rng_key, 30, n, **PRIOR_KWARGS,
            )
            assert data.shape == (n, 30, 2)
            assert reconstruct_log_theta(context).shape == (n, NUM_PARAMS)
