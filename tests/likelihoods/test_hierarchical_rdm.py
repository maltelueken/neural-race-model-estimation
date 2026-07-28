"""Tests for the hierarchical RDM likelihood.

The hierarchical likelihood is a function of subject-level log-parameters only,
shape ``(S, 5)``. Population-level parameters enter the posterior through the
prior, not here; the caller reconstructs ``log_theta`` from them before calling.
"""

import jax
import jax.numpy as jnp

from confrdm_jax.likelihoods import (
    create_rdm_hierarchical_likelihood,
    create_rdm_two_accumulators_likelihood,
)


NUM_PARAMS = 5

# log of [v_intercept, v_slope, s_true, b, t0] = [1.0, 1.5, 1.2, 1.2, 0.3]
MU = jnp.array([0.0, 0.4, 0.2, 0.2, -1.2])


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


def _make_log_theta(key, num_subjects=3):
    """Plausible subject-level log-parameters, shape (S, 5)."""
    log_theta = jnp.tile(MU, (num_subjects, 1))
    return log_theta + 0.05 * jax.random.normal(key, (num_subjects, NUM_PARAMS))


class TestRdmHierarchicalLikelihood:
    """Tests for create_rdm_hierarchical_likelihood."""

    def test_output_is_finite_scalar(self):
        k1, k2 = jax.random.split(jax.random.PRNGKey(42))
        data, mask, _ = _make_synthetic_data(k1)
        log_theta = _make_log_theta(k2)

        result = create_rdm_hierarchical_likelihood(data, mask)(log_theta)

        assert result.shape == ()
        assert jnp.isfinite(result)

    def test_masked_trials_do_not_contribute(self):
        """Likelihood should not change when padded values differ."""
        k1, k2 = jax.random.split(jax.random.PRNGKey(123))
        data, mask, _ = _make_synthetic_data(k1, trial_counts=[20, 10, 15])
        log_theta = _make_log_theta(k2)

        result1 = create_rdm_hierarchical_likelihood(data, mask)(log_theta)

        # Modify padded region with garbage values
        data_modified = data.at[1, 10:, :].set(999.0)
        result2 = create_rdm_hierarchical_likelihood(data_modified, mask)(log_theta)

        assert jnp.allclose(result1, result2)

    def test_matches_single_subject_likelihood(self):
        """For a single subject, the hierarchical sum should match the reference."""
        k1, k2 = jax.random.split(jax.random.PRNGKey(7))

        n_trials = 40
        rts = 0.3 + 0.5 * jax.random.uniform(k1, (n_trials,))
        choices = jax.random.bernoulli(k2, 0.6, (n_trials,)).astype(float)
        single_data = jnp.stack([rts, choices], axis=-1)

        subj_params_log = jnp.log(jnp.array([1.0, 1.5, 1.2, 1.2, 0.3]))
        single_result = jnp.sum(
            create_rdm_two_accumulators_likelihood(single_data)(subj_params_log)
        )

        data = single_data[None, :, :]  # (1, n_trials, 2)
        mask = jnp.ones((1, n_trials), dtype=bool)
        hier_result = create_rdm_hierarchical_likelihood(data, mask)(
            subj_params_log[None, :]
        )

        assert jnp.allclose(single_result, hier_result, atol=1e-5)

    def test_sums_over_subjects(self):
        """Total should equal the sum of the per-subject reference likelihoods."""
        k1, k2 = jax.random.split(jax.random.PRNGKey(5))
        trial_counts = [30, 30, 30]
        data, mask, _ = _make_synthetic_data(k1, trial_counts=trial_counts)
        log_theta = _make_log_theta(k2)

        total = create_rdm_hierarchical_likelihood(data, mask)(log_theta)

        expected = sum(
            jnp.sum(create_rdm_two_accumulators_likelihood(data[s])(log_theta[s]))
            for s in range(len(trial_counts))
        )

        assert jnp.allclose(total, expected, atol=1e-5)

    def test_more_subjects_changes_likelihood(self):
        """Adding subjects should change the total likelihood."""
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(99), 3)

        data_2, mask_2, _ = _make_synthetic_data(k1, num_subjects=2, trial_counts=[30, 30])
        data_3, mask_3, _ = _make_synthetic_data(k2, num_subjects=3, trial_counts=[30, 30, 30])

        log_theta_2 = _make_log_theta(k3, num_subjects=2)
        # Extend to 3 subjects by repeating the last subject's parameters
        log_theta_3 = jnp.concatenate([log_theta_2, log_theta_2[-1:]], axis=0)

        ll_2 = create_rdm_hierarchical_likelihood(data_2, mask_2)(log_theta_2)
        ll_3 = create_rdm_hierarchical_likelihood(data_3, mask_3)(log_theta_3)

        assert not jnp.allclose(ll_2, ll_3)

    def test_t0_above_every_rt_is_penalised(self):
        """rt <= t0 is impossible; the penalty must stay gradient-carrying."""
        data, mask, _ = _make_synthetic_data(jax.random.PRNGKey(11))
        ll_fn = create_rdm_hierarchical_likelihood(data, mask)

        log_theta = jnp.tile(MU, (3, 1))
        # t0 = e^0.5 ~ 1.65 s, above every simulated RT (which top out at 0.8 s)
        bad = ll_fn(log_theta.at[:, 4].set(0.5))
        good = ll_fn(log_theta)

        assert bad < good
        grad = jax.grad(lambda lt: ll_fn(lt))(log_theta.at[:, 4].set(0.5))
        # Gradient must push t0 down, not sit at a flat floor
        assert jnp.all(grad[:, 4] < 0)
