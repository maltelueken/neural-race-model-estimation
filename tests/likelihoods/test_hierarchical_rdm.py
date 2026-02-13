"""Tests for hierarchical RDM likelihood and prior."""

import jax
import jax.numpy as jnp

from confrdm_jax.likelihoods import (
    create_rdm_hierarchical_likelihood,
    create_rdm_two_accumulators_likelihood,
)


NUM_PARAMS = 5
P = 5
# MVN prior layout: log_psi (5) + mu (5) + L_flat (15) = 25 population params
NUM_PSI = P
NUM_MU = P
NUM_L_DIAG = P
NUM_L_OFFDIAG = P * (P - 1) // 2
NUM_L_PARAMS = NUM_L_DIAG + NUM_L_OFFDIAG
NUM_POP_PARAMS = NUM_PSI + NUM_MU + NUM_L_PARAMS  # 5 + 5 + 15 = 25


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
    """Create a plausible hierarchical parameter vector.

    Layout for MVN prior: [log_psi (5), mu (5), log_diag_L (5), offdiag_L (10), log_theta (S*5)]
    """
    k1, k2 = jax.random.split(key)

    # InverseGamma scale params (log-transformed for positivity)
    log_psi = jnp.zeros(NUM_PSI)  # psi = 1.0

    # Population mean (unconstrained, in log-space for positive params)
    mu = jnp.array([0.0, 0.4, 0.2, 0.2, -1.2])  # log of [1.0, 1.5, 1.2, 1.2, 0.3]

    # Cholesky factor: log-diagonal + off-diagonal
    log_diag_L = jnp.array([-0.5, -0.5, -0.5, -0.5, -0.5])  # small variances
    offdiag_L = jnp.zeros(NUM_L_OFFDIAG)  # no correlations for simplicity
    L_flat = jnp.concatenate([log_diag_L, offdiag_L])

    # Subject params in log-space (log_theta)
    log_theta = jnp.tile(mu, (num_subjects, 1))
    log_theta = log_theta + 0.05 * jax.random.normal(k1, (num_subjects, NUM_PARAMS))

    x = jnp.concatenate([log_psi, mu, L_flat, log_theta.ravel()])
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
        pop_params = jnp.zeros(NUM_POP_PARAMS)
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