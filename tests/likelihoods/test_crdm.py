"""Tests for approximate CRDM likelihoods."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from confrdm_jax.flows import make_mlp_conditioner
from confrdm_jax.likelihoods import create_crdm_likelihood_factory_approx


@pytest.fixture
def crdm_conditioner():
    """Create an untrained CRDM conditioner for testing (5 context params: v_c, amp, tau, s, b)."""
    rngs = nnx.Rngs(default=1)
    cond = make_mlp_conditioner(num_in=5, num_bins=4, num_mid=64, rngs=rngs)
    cond.eval()
    return cond


def _make_crdm_mock_data(key, n_trials=50):
    """Create synthetic CRDM data, shape (n_trials, 3) with columns [RT, choice, condition]."""
    k1, k2, k3 = jax.random.split(key, 3)
    rts = 0.3 + 0.5 * jax.random.uniform(k1, (n_trials,))
    choices = jax.random.bernoulli(k2, 0.6, (n_trials,)).astype(float)
    conditions = jax.random.bernoulli(k3, 0.5, (n_trials,)).astype(float)
    return jnp.stack([rts, choices, conditions], axis=-1)


# CRDM has 7 params: v_c_intercept, v_c_slope, amp, tau, s_true, b, t0

@pytest.fixture
def crdm_params(): return jnp.log(jnp.array([1.0, 4.0, 0.3, 0.1, 0.8, 0.7, 0.3]))

@pytest.fixture
def crdm_params_alt(): return jnp.log(jnp.array([1.5, 2.0, 0.5, 0.2, 1.0, 1.0, 0.2]))


class TestCrdmLikelihoodFactoryApprox:
    """Tests for create_crdm_likelihood_factory_approx."""

    def test_output_shape_and_finiteness(self, crdm_conditioner, crdm_params):
        """Likelihood should return a per-trial array of finite values."""
        data = _make_crdm_mock_data(jax.random.PRNGKey(10))
        create_likelihood = create_crdm_likelihood_factory_approx(crdm_conditioner)
        ll_fn = create_likelihood(data)

        result = ll_fn(crdm_params)

        assert result.shape == (data.shape[0],)
        assert jnp.all(jnp.isfinite(result))

    def test_output_is_negative(self, crdm_conditioner, crdm_params):
        """Most log-likelihoods should be negative."""
        data = _make_crdm_mock_data(jax.random.PRNGKey(11), n_trials=200)
        create_likelihood = create_crdm_likelihood_factory_approx(crdm_conditioner)
        ll_fn = create_likelihood(data)

        result = ll_fn(crdm_params)

        assert jnp.mean(result < 0) > 0.5

    def test_different_params_give_different_likelihoods(self, crdm_conditioner, crdm_params):
        """Different parameter vectors should produce different likelihoods."""
        data = _make_crdm_mock_data(jax.random.PRNGKey(12))
        create_likelihood = create_crdm_likelihood_factory_approx(crdm_conditioner)
        ll_fn = create_likelihood(data)

        crdm_params_alt = jnp.log(jnp.array([1.5, 2.0, 0.5, 0.2, 1.0, 1.0, 0.2]))

        result1 = ll_fn(crdm_params)
        result2 = ll_fn(crdm_params_alt)

        assert not jnp.allclose(result1, result2)

    def test_different_data_give_different_likelihoods(self, crdm_conditioner, crdm_params):
        """Different datasets should produce different likelihoods for the same params."""
        data1 = _make_crdm_mock_data(jax.random.PRNGKey(13))
        data2 = _make_crdm_mock_data(jax.random.PRNGKey(14))
        create_likelihood = create_crdm_likelihood_factory_approx(crdm_conditioner)

        ll_fn1 = create_likelihood(data1)
        ll_fn2 = create_likelihood(data2)

        assert not jnp.allclose(jnp.sum(ll_fn1(crdm_params)), jnp.sum(ll_fn2(crdm_params)))

    def test_factory_returns_callable(self, crdm_conditioner):
        """The factory should return a create_likelihood function, which returns a likelihood_fun."""
        create_likelihood = create_crdm_likelihood_factory_approx(crdm_conditioner)
        assert callable(create_likelihood)

        data = _make_crdm_mock_data(jax.random.PRNGKey(15))
        ll_fn = create_likelihood(data)
        assert callable(ll_fn)

    def test_congruent_vs_incongruent_differ(self, crdm_conditioner, crdm_params):
        """All-congruent vs all-incongruent data should yield different likelihoods."""
        key = jax.random.PRNGKey(16)
        k1, k2 = jax.random.split(key)
        n_trials = 50

        rts = 0.3 + 0.5 * jax.random.uniform(k1, (n_trials,))
        choices = jax.random.bernoulli(k2, 0.6, (n_trials,)).astype(float)

        data_cong = jnp.stack([rts, choices, jnp.ones(n_trials)], axis=-1)
        data_incong = jnp.stack([rts, choices, jnp.zeros(n_trials)], axis=-1)

        create_likelihood = create_crdm_likelihood_factory_approx(crdm_conditioner)
        ll_cong = create_likelihood(data_cong)
        ll_incong = create_likelihood(data_incong)

        assert not jnp.allclose(jnp.sum(ll_cong(crdm_params)), jnp.sum(ll_incong(crdm_params)))


class TestCrdmCensoring:
    """Right-censoring of the ``rt = -1.0`` sentinel in the CRDM likelihoods.

    A trial that never crossed is the observation ``T > t_max``, whose racing
    likelihood is ``S_true(t_max) * S_false(t_max)``.  Without `t_max` the
    sentinel falls into the ``rt <= t0`` branch of ``_penalize_invalid_rt``
    instead and contributes ~-1300 with a gradient driving ``t0`` to zero.
    """

    T_MAX = 4.0

    def _censored(self, key, n_trials=20, n_censored=3):
        data = _make_crdm_mock_data(key, n_trials)
        return data.at[:n_censored, 0].set(-1.0).at[:n_censored, 1].set(-1.0)

    def test_no_sentinel_is_bit_identical(self, crdm_conditioner, crdm_params):
        """With no censored trials, passing t_max must change nothing at all."""
        data = _make_crdm_mock_data(jax.random.PRNGKey(20))
        off = create_crdm_likelihood_factory_approx(crdm_conditioner)(data)
        on = create_crdm_likelihood_factory_approx(
            crdm_conditioner, t_max=self.T_MAX,
        )(data)
        assert jnp.array_equal(off(crdm_params), on(crdm_params))

    def test_censored_trials_scored_by_survival(self, crdm_conditioner, crdm_params):
        """Censored trials get log S_true(t_max) + log S_false(t_max), not the penalty."""
        data = self._censored(jax.random.PRNGKey(21))
        off = create_crdm_likelihood_factory_approx(crdm_conditioner)(data)(crdm_params)
        on = create_crdm_likelihood_factory_approx(
            crdm_conditioner, t_max=self.T_MAX,
        )(data)(crdm_params)

        # Uncensored trials are untouched; censored ones leave the penalty regime.
        assert jnp.array_equal(off[3:], on[3:])
        assert jnp.all(off[:3] < -1000.0)
        assert jnp.all(on[:3] > -1000.0)
        # A survival probability is still a probability.
        assert jnp.all(on[:3] < 0.0)

    def test_censored_trial_has_no_t0_gradient(self, crdm_conditioner, crdm_params):
        """A censored trial must contribute exactly zero gradient in ``t0``.

        The horizon is on the *decision*-time scale — the simulator integrates
        for ``t_max`` and only then adds ``t0`` — so "no crossing within
        ``t_max``" carries no information about ``t0`` at all. Contrast the
        unhandled sentinel, whose slope is ``-1e3`` per trial and is what
        collapses window adaptation.
        """
        data = self._censored(jax.random.PRNGKey(22), n_trials=10, n_censored=10)

        def total(params, t_max):
            fn = create_crdm_likelihood_factory_approx(crdm_conditioner, t_max=t_max)
            return jnp.sum(fn(data)(params))

        g_off = jax.grad(total)(crdm_params, None)[6]
        g_on = jax.grad(total)(crdm_params, self.T_MAX)[6]

        # -1e3 per trial, times t0 for the log-space chain rule, times 10 trials.
        assert jnp.isclose(g_off, -1e3 * jnp.exp(crdm_params[6]) * 10, rtol=1e-4)
        assert g_on == 0.0

    def test_hierarchical_censoring_is_per_subject(self, crdm_conditioner):
        """Only the subjects that censor should have their own t0 gradient distorted."""
        from confrdm_jax.likelihoods import (
            create_crdm_hierarchical_likelihood_factory_approx,
        )

        n_subjects, n_trials = 3, 20
        keys = jax.random.split(jax.random.PRNGKey(23), n_subjects)
        data = jnp.stack([_make_crdm_mock_data(k, n_trials) for k in keys])
        # Subject 2 alone censors one trial.
        data = data.at[2, 0, 0].set(-1.0).at[2, 0, 1].set(-1.0)
        mask = jnp.ones((n_subjects, n_trials), dtype=bool)
        theta = jnp.log(
            jnp.tile(jnp.array([1.0, 4.0, 0.3, 0.1, 0.8, 0.7, 0.3]), (n_subjects, 1)),
        )

        # Masking the censored trial out is the reference: since a censored
        # trial carries no t0 information, keeping it must give the *same* t0
        # gradient as dropping it. Comparing subject 2 against its siblings
        # would confound this with its having one fewer scored trial.
        mask_drop = mask.at[2, 0].set(False)

        def grad_t0(t_max, m):
            fn = create_crdm_hierarchical_likelihood_factory_approx(
                crdm_conditioner, t_max=t_max,
            )(data, m)
            return jax.grad(lambda v: fn(theta.at[:, 6].set(v)))(theta[:, 6])

        # Handled: keeping the sentinel is equivalent to dropping it, for t0.
        assert jnp.allclose(
            grad_t0(self.T_MAX, mask), grad_t0(self.T_MAX, mask_drop), atol=1e-5,
        )
        # Unhandled: it drags subject 2's own t0 and no one else's.
        g_off, g_off_drop = grad_t0(None, mask), grad_t0(None, mask_drop)
        assert jnp.allclose(g_off[:2], g_off_drop[:2], atol=1e-5)
        assert jnp.abs(g_off[2] - g_off_drop[2]) > 100.0
