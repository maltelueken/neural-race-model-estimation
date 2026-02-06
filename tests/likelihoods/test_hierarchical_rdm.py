"""Tests for hierarchical RDM likelihood and prior."""

import jax
import jax.numpy as jnp

from confrdm_jax.likelihoods import (
    create_rdm_hierarchical_likelihood,
    create_rdm_two_accumulators_likelihood,
)


NUM_PARAMS = 5


def _make_synthetic_data(key, num_subjects=3, trial_counts=None):
    """Create padded synthetic data and mask for testing.

    Returns:
        data: shape (S, max_trials, 2)
        mask: shape (S, max_trials)
        trial_counts: list of ints
    """
    if trial_counts is None:
        trial_counts = [50, 30, 40]
    assert len(trial_counts) == num_subjects

    max_trials = max(trial_counts)
    data = jnp.zeros((num_subjects, max_trials, 2))
    mask = jnp.zeros((num_subjects, max_trials), dtype=bool)

    for s, n in enumerate(trial_counts):
        key, k1, k2 = jax.random.split(key, 3)
        rts = 0.3 + 0.5 * jax.random.uniform(k1, (n,))
        choices = jax.random.bernoulli(k2, 0.6, (n,)).astype(float)
        trials = jnp.stack([rts, choices], axis=-1)
        data = data.at[s, :n, :].set(trials)
        mask = mask.at[s, :n].set(True)

    return data, mask, trial_counts


def _make_param_vector(key, num_subjects=3):
    """Create a plausible hierarchical parameter vector in log-space."""
    # Population params: mu and sigma for each of the 5 RDM params
    # Reasonable RDM params (in natural space): v_intercept~1, v_slope~1.5, s_true~1.2, b~1.2, t0~0.3
    pop_mu = jnp.array([1.0, 1.5, 1.2, 1.2, 0.3])
    pop_sigma = jnp.array([0.1, 0.2, 0.1, 0.1, 0.05])

    # Interleave: [mu_1, sigma_1, mu_2, sigma_2, ...]
    pop_params = jnp.zeros(2 * NUM_PARAMS)
    pop_params = pop_params.at[0::2].set(pop_mu)
    pop_params = pop_params.at[1::2].set(pop_sigma)

    # Subject params drawn near population means
    subj_params = jnp.tile(pop_mu, (num_subjects, 1))
    subj_params = subj_params + 0.05 * jax.random.normal(key, (num_subjects, NUM_PARAMS))
    subj_params = jnp.maximum(subj_params, 0.01)  # keep positive before log

    x = jnp.concatenate([jnp.log(pop_params), jnp.log(subj_params.ravel())])
    return x


class TestRdmHierarchicalLikelihood:
    """Tests for create_rdm_hierarchical_likelihood."""

    def test_output_is_finite_scalar(self):
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        data, mask, _ = _make_synthetic_data(k1)
        x = _make_param_vector(k2)

        ll_fn = create_rdm_hierarchical_likelihood(data, mask)
        result = ll_fn(x)

        assert result.shape == ()
        assert jnp.isfinite(result)

    def test_masked_trials_do_not_contribute(self):
        """Likelihood should not change when padded values differ."""
        key = jax.random.PRNGKey(123)
        k1, k2, k3 = jax.random.split(key, 3)
        data, mask, _ = _make_synthetic_data(k1, trial_counts=[20, 10, 15])
        x = _make_param_vector(k2)

        ll_fn = create_rdm_hierarchical_likelihood(data, mask)
        result1 = ll_fn(x)

        # Modify padded region with garbage values
        data_modified = data.at[1, 10:, :].set(999.0)
        ll_fn2 = create_rdm_hierarchical_likelihood(data_modified, mask)
        result2 = ll_fn2(x)

        assert jnp.allclose(result1, result2)

    def test_matches_single_subject_likelihood(self):
        """For a single subject, hierarchical likelihood should match the original."""
        key = jax.random.PRNGKey(7)
        k1, k2 = jax.random.split(key)

        n_trials = 40
        rts = 0.3 + 0.5 * jax.random.uniform(k1, (n_trials,))
        choices = jax.random.bernoulli(k2, 0.6, (n_trials,)).astype(float)
        single_data = jnp.stack([rts, choices], axis=-1)

        # Single-subject likelihood
        single_ll_fn = create_rdm_two_accumulators_likelihood(single_data)
        subj_params_log = jnp.log(jnp.array([1.0, 1.5, 1.2, 1.2, 0.3]))
        single_result = jnp.sum(single_ll_fn(subj_params_log))

        # Hierarchical with 1 subject
        data = single_data[None, :, :]  # (1, n_trials, 2)
        mask = jnp.ones((1, n_trials), dtype=bool)

        # Population params don't affect likelihood, only prior
        pop_params = jnp.zeros(2 * NUM_PARAMS)
        x = jnp.concatenate([pop_params, subj_params_log])

        hier_ll_fn = create_rdm_hierarchical_likelihood(data, mask)
        hier_result = hier_ll_fn(x)

        assert jnp.allclose(single_result, hier_result, atol=1e-5)

    def test_more_subjects_changes_likelihood(self):
        """Adding subjects should change the total likelihood."""
        key = jax.random.PRNGKey(99)
        k1, k2, k3 = jax.random.split(key, 3)

        data_2, mask_2, _ = _make_synthetic_data(k1, num_subjects=2, trial_counts=[30, 30])
        data_3, mask_3, _ = _make_synthetic_data(k2, num_subjects=3, trial_counts=[30, 30, 30])

        x_2 = _make_param_vector(k3, num_subjects=2)
        # Extend to 3 subjects by repeating last subject's params
        x_3 = jnp.concatenate([x_2, x_2[-NUM_PARAMS:]])

        ll_2 = create_rdm_hierarchical_likelihood(data_2, mask_2)(x_2)
        ll_3 = create_rdm_hierarchical_likelihood(data_3, mask_3)(x_3)

        assert not jnp.allclose(ll_2, ll_3)