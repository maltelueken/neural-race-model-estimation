"""Tests for hierarchical RDM simulator and prior."""

import jax
import jax.numpy as jnp

from confrdm_jax.simulators import (
    create_rdm_hierarchical_prior,
    sample_conditional_rdm_hierarchical,
)


NUM_PARAMS = 5


class TestCreateRdmHierarchicalPrior:
    """Tests for create_rdm_hierarchical_prior."""

    def test_sample_has_five_param_groups(self, rng_key):
        prior = create_rdm_hierarchical_prior(num_subjects=3)
        sample = prior.sample(seed=rng_key)
        assert len(sample) == NUM_PARAMS

    def test_truncated_normal_hyperprior_structure(self, rng_key):
        """Truncated normal groups (v_intercept, v_slope, t0) should have (mu, sigma, subjects)."""
        prior = create_rdm_hierarchical_prior(num_subjects=4)
        sample = prior.sample(seed=rng_key)

        for i in [0, 1, 4]:  # v_intercept, v_slope, t0
            assert len(sample[i]) == 3
            assert sample[i][0].shape == ()    # mu
            assert sample[i][1].shape == ()    # sigma
            assert sample[i][2].shape == (4,)  # subject values

    def test_gamma_hyperprior_structure(self, rng_key):
        """Gamma groups (s_true, b) should have (scale, subjects)."""
        prior = create_rdm_hierarchical_prior(num_subjects=4)
        sample = prior.sample(seed=rng_key)

        for i in [2, 3]:  # s_true, b
            assert len(sample[i]) == 2
            assert sample[i][0].shape == ()    # scale
            assert sample[i][1].shape == (4,)  # subject values

    def test_all_samples_positive(self, rng_key):
        """All sampled values should be positive."""
        prior = create_rdm_hierarchical_prior(num_subjects=5)
        sample = prior.sample(seed=rng_key)

        for group in sample:
            for val in group:
                assert jnp.all(val > 0)

    def test_log_prob_is_finite(self, rng_key):
        prior = create_rdm_hierarchical_prior(num_subjects=3)
        sample = prior.sample(seed=rng_key)
        lp = prior.log_prob(sample)
        assert lp.shape == ()
        assert jnp.isfinite(lp)

    def test_num_subjects_changes_sample_shape(self, rng_key):
        prior_2 = create_rdm_hierarchical_prior(num_subjects=2)
        prior_7 = create_rdm_hierarchical_prior(num_subjects=7)

        sample_2 = prior_2.sample(seed=rng_key)
        sample_7 = prior_7.sample(seed=rng_key)

        assert sample_2[0][-1].shape == (2,)
        assert sample_7[0][-1].shape == (7,)


class TestSampleConditionalRdmHierarchical:
    """Tests for sample_conditional_rdm_hierarchical."""

    def test_data_shape(self, rng_key):
        num_subjects, num_trials = 3, 100
        prior = create_rdm_hierarchical_prior(num_subjects)
        data, _ = sample_conditional_rdm_hierarchical(rng_key, num_trials, prior)

        assert data.shape == (num_subjects, num_trials, 2)

    def test_data_is_finite(self, rng_key):
        prior = create_rdm_hierarchical_prior(num_subjects=3)
        data, _ = sample_conditional_rdm_hierarchical(rng_key, 50, prior)

        assert jnp.all(jnp.isfinite(data))

    def test_rt_greater_than_t0(self, rng_key):
        """All RTs should be greater than the subject's t0."""
        prior = create_rdm_hierarchical_prior(num_subjects=3)
        data, context = sample_conditional_rdm_hierarchical(rng_key, 200, prior)

        t0_per_subject = context[4][-1]  # shape (S,)
        for s in range(3):
            assert jnp.all(data[s, :, 0] > t0_per_subject[s])

    def test_choices_are_binary(self, rng_key):
        prior = create_rdm_hierarchical_prior(num_subjects=3)
        data, _ = sample_conditional_rdm_hierarchical(rng_key, 100, prior)

        choices = data[:, :, 1]
        assert jnp.all((choices == 0) | (choices == 1))

    def test_context_structure_matches_prior(self, rng_key):
        """Context should have the same nested structure as the prior sample."""
        num_subjects = 4
        prior = create_rdm_hierarchical_prior(num_subjects)
        _, context = sample_conditional_rdm_hierarchical(rng_key, 50, prior)

        assert len(context) == NUM_PARAMS
        # Truncated normal groups
        for i in [0, 1, 4]:
            assert len(context[i]) == 3
            assert context[i][2].shape == (num_subjects,)
        # Gamma groups
        for i in [2, 3]:
            assert len(context[i]) == 2
            assert context[i][1].shape == (num_subjects,)

    def test_context_has_valid_log_prob(self, rng_key):
        """The returned context should be a valid sample under the prior."""
        prior = create_rdm_hierarchical_prior(num_subjects=3)
        _, context = sample_conditional_rdm_hierarchical(rng_key, 50, prior)

        lp = prior.log_prob(context)
        assert jnp.isfinite(lp)

    def test_reproducibility(self, rng_key):
        """Same key should produce identical data and context."""
        prior = create_rdm_hierarchical_prior(num_subjects=3)

        data1, ctx1 = sample_conditional_rdm_hierarchical(rng_key, 50, prior)
        data2, ctx2 = sample_conditional_rdm_hierarchical(rng_key, 50, prior)

        assert jnp.allclose(data1, data2)
        for g1, g2 in zip(ctx1, ctx2):
            for v1, v2 in zip(g1, g2):
                assert jnp.allclose(v1, v2)

    def test_different_keys_produce_different_data(self, rng_key):
        prior = create_rdm_hierarchical_prior(num_subjects=3)
        key2 = jax.random.split(rng_key)[0]

        data1, _ = sample_conditional_rdm_hierarchical(rng_key, 100, prior)
        data2, _ = sample_conditional_rdm_hierarchical(key2, 100, prior)

        assert not jnp.allclose(data1, data2)

    def test_different_num_subjects(self, rng_key):
        """Different subject counts should produce correctly shaped data."""
        for n in [1, 2, 5, 10]:
            prior = create_rdm_hierarchical_prior(num_subjects=n)
            data, context = sample_conditional_rdm_hierarchical(rng_key, 30, prior)
            assert data.shape == (n, 30, 2)
            assert context[0][-1].shape == (n,)
