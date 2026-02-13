"""Tests for hierarchical RDM simulator and LKJ-MVN prior."""

import jax
import jax.numpy as jnp

from confrdm_jax.simulators import (
    create_hierarchical_rdm_prior_lkj_mvn,
    sample_conditional_rdm_hierarchical_lkj_mvn,
)


NUM_PARAMS = 5


class TestCreateHierarchicalRdmPriorLkjMvn:
    """Tests for create_hierarchical_rdm_prior_lkj_mvn."""

    def test_sample_returns_four_components(self, rng_key):
        """Sample should return (s, L, mu, log_theta)."""
        prior = create_hierarchical_rdm_prior_lkj_mvn(num_subjects=3)
        sample = prior.sample(seed=rng_key)
        assert len(sample) == 4

    def test_sample_shapes(self, rng_key):
        """Check shapes of (s, L, mu, log_theta)."""
        num_subjects = 4
        prior = create_hierarchical_rdm_prior_lkj_mvn(num_subjects=num_subjects)
        s, L, mu, log_theta = prior.sample(seed=rng_key)

        assert s.shape == (NUM_PARAMS,)
        assert L.shape == (NUM_PARAMS, NUM_PARAMS)
        assert mu.shape == (NUM_PARAMS,)
        assert log_theta.shape == (num_subjects, NUM_PARAMS)

    def test_s_positive(self, rng_key):
        """Standard deviations should be positive (HalfNormal)."""
        prior = create_hierarchical_rdm_prior_lkj_mvn(num_subjects=3)
        s, _, _, _ = prior.sample(seed=rng_key)
        assert jnp.all(s > 0)

    def test_L_lower_triangular(self, rng_key):
        """Cholesky factor L should be lower triangular."""
        prior = create_hierarchical_rdm_prior_lkj_mvn(num_subjects=3)
        _, L, _, _ = prior.sample(seed=rng_key)
        assert jnp.allclose(L, jnp.tril(L))

    def test_log_prob_is_finite(self, rng_key):
        prior = create_hierarchical_rdm_prior_lkj_mvn(num_subjects=3)
        _, L, mu, log_theta = prior.sample(seed=rng_key)
        lp = prior.log_prob(L, mu, log_theta)
        assert jnp.all(jnp.isfinite(lp))

    def test_num_subjects_changes_log_theta_shape(self, rng_key):
        prior_2 = create_hierarchical_rdm_prior_lkj_mvn(num_subjects=2)
        prior_7 = create_hierarchical_rdm_prior_lkj_mvn(num_subjects=7)

        _, _, _, log_theta_2 = prior_2.sample(seed=rng_key)
        _, _, _, log_theta_7 = prior_7.sample(seed=rng_key)

        assert log_theta_2.shape == (2, NUM_PARAMS)
        assert log_theta_7.shape == (7, NUM_PARAMS)


class TestSampleConditionalRdmHierarchicalLkjMvn:
    """Tests for sample_conditional_rdm_hierarchical_lkj_mvn."""

    def test_data_shape(self, rng_key):
        num_subjects, num_trials = 3, 100
        data, _ = sample_conditional_rdm_hierarchical_lkj_mvn(
            rng_key, num_trials, num_subjects,
        )
        assert data.shape == (num_subjects, num_trials, 2)

    def test_data_is_finite(self, rng_key):
        data, _ = sample_conditional_rdm_hierarchical_lkj_mvn(
            rng_key, 50, num_subjects=3,
        )
        assert jnp.all(jnp.isfinite(data))

    def test_rt_greater_than_t0(self, rng_key):
        """All RTs should be greater than the subject's t0."""
        num_subjects = 3
        data, context = sample_conditional_rdm_hierarchical_lkj_mvn(
            rng_key, 200, num_subjects,
        )
        t0_per_subject = context['theta'][:, 4]  # shape (S,)
        for s in range(num_subjects):
            assert jnp.all(data[s, :, 0] > t0_per_subject[s])

    def test_choices_are_binary(self, rng_key):
        data, _ = sample_conditional_rdm_hierarchical_lkj_mvn(
            rng_key, 100, num_subjects=3,
        )
        choices = data[:, :, 1]
        assert jnp.all((choices == 0) | (choices == 1))

    def test_context_keys(self, rng_key):
        """Context should be a dict with expected keys."""
        num_subjects = 4
        _, context = sample_conditional_rdm_hierarchical_lkj_mvn(
            rng_key, 50, num_subjects,
        )
        expected_keys = {'rho', 's', 'mu', 'L', 'Sigma', 'log_theta', 'theta'}
        assert set(context.keys()) == expected_keys

    def test_context_shapes(self, rng_key):
        """Context arrays should have correct shapes."""
        num_subjects = 4
        _, context = sample_conditional_rdm_hierarchical_lkj_mvn(
            rng_key, 50, num_subjects,
        )
        P = NUM_PARAMS
        assert context['rho'].shape == (P, P)
        assert context['s'].shape == (P,)
        assert context['mu'].shape == (P,)
        assert context['L'].shape == (P, P)
        assert context['Sigma'].shape == (P, P)
        assert context['log_theta'].shape == (num_subjects, P)
        assert context['theta'].shape == (num_subjects, P)

    def test_context_has_valid_log_prob(self, rng_key):
        """The returned context should be a valid sample under the prior."""
        num_subjects = 3
        prior = create_hierarchical_rdm_prior_lkj_mvn(num_subjects)
        _, context = sample_conditional_rdm_hierarchical_lkj_mvn(
            rng_key, 50, num_subjects,
        )
        lp = prior.log_prob(context['L'], context['mu'], context['log_theta'])
        assert jnp.all(jnp.isfinite(lp))

    def test_reproducibility(self, rng_key):
        """Same key should produce identical data and context."""
        num_subjects = 3
        data1, ctx1 = sample_conditional_rdm_hierarchical_lkj_mvn(
            rng_key, 50, num_subjects,
        )
        data2, ctx2 = sample_conditional_rdm_hierarchical_lkj_mvn(
            rng_key, 50, num_subjects,
        )
        assert jnp.allclose(data1, data2)
        for k in ctx1:
            assert jnp.allclose(ctx1[k], ctx2[k])

    def test_different_keys_produce_different_data(self, rng_key):
        num_subjects = 3
        key2 = jax.random.split(rng_key)[0]

        data1, _ = sample_conditional_rdm_hierarchical_lkj_mvn(
            rng_key, 100, num_subjects,
        )
        data2, _ = sample_conditional_rdm_hierarchical_lkj_mvn(
            key2, 100, num_subjects,
        )
        assert not jnp.allclose(data1, data2)

    def test_different_num_subjects(self, rng_key):
        """Different subject counts should produce correctly shaped data."""
        for n in [1, 2, 5, 10]:
            data, context = sample_conditional_rdm_hierarchical_lkj_mvn(
                rng_key, 30, n,
            )
            assert data.shape == (n, 30, 2)
            assert context['log_theta'].shape == (n, NUM_PARAMS)
