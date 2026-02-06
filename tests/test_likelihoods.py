"""Tests for approximate likelihoods."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from confrdm_jax.flows import make_mlp_conditioner
from confrdm_jax.likelihoods import (
    create_crdm_likelihood_factory_approx,
    create_rdm_likelihood_factory_approx,
    create_rdm_two_accumulators_likelihood,
)


@pytest.fixture
def rdm_conditioner():
    """Create an untrained RDM conditioner for testing (3 context params: v, s, b)."""
    rngs = nnx.Rngs(default=0)
    cond = make_mlp_conditioner(num_in=3, num_bins=4, num_mid=64, rngs=rngs)
    cond.eval()
    return cond


@pytest.fixture
def crdm_conditioner():
    """Create an untrained CRDM conditioner for testing (5 context params: v_c, amp, tau, s, b)."""
    rngs = nnx.Rngs(default=1)
    cond = make_mlp_conditioner(num_in=5, num_bins=4, num_mid=64, rngs=rngs)
    cond.eval()
    return cond

@pytest.fixture
def rdm_params(): return jnp.log(jnp.array([1.0, 1.5, 1.2, 1.2, 0.3]))

def _make_rdm_mock_data(key, n_trials=50):
    """Create synthetic single-subject data, shape (n_trials, 2)."""
    k1, k2 = jax.random.split(key)
    rts = 0.3 + 0.5 * jax.random.uniform(k1, (n_trials,))
    choices = jax.random.bernoulli(k2, 0.6, (n_trials,)).astype(float)
    return jnp.stack([rts, choices], axis=-1)


class TestRdmLikelihoodFactoryApprox:
    """Tests for create_rdm_likelihood_factory_approx."""

    def test_output_shape_and_finiteness(self, rdm_conditioner, rdm_params):
        """Likelihood should return a per-trial array of finite values."""
        data = _make_rdm_mock_data(jax.random.PRNGKey(0))
        create_likelihood = create_rdm_likelihood_factory_approx(rdm_conditioner)
        ll_fn = create_likelihood(data)

        result = ll_fn(rdm_params)

        assert result.shape == (data.shape[0],)
        assert jnp.all(jnp.isfinite(result))

    def test_output_is_negative(self, rdm_conditioner, rdm_params):
        """Most log-likelihoods should be negative."""
        data = _make_rdm_mock_data(jax.random.PRNGKey(1), n_trials=200)
        create_likelihood = create_rdm_likelihood_factory_approx(rdm_conditioner)
        ll_fn = create_likelihood(data)

        result = ll_fn(rdm_params)

        assert jnp.mean(result < 0) > 0.5

    def test_different_params_give_different_likelihoods(self, rdm_conditioner, rdm_params):
        """Different parameter vectors should produce different likelihoods."""
        data = _make_rdm_mock_data(jax.random.PRNGKey(2))
        create_likelihood = create_rdm_likelihood_factory_approx(rdm_conditioner)
        ll_fn = create_likelihood(data)

        rdm_params_alt = jnp.log(jnp.array([2.0, 0.5, 0.8, 1.5, 0.2]))

        result = ll_fn(rdm_params)
        result_alt = ll_fn(rdm_params_alt)

        assert not jnp.allclose(result, result_alt)

    def test_different_data_give_different_likelihoods(self, rdm_conditioner, rdm_params):
        """Different datasets should produce different likelihoods for the same params."""
        data1 = _make_rdm_mock_data(jax.random.PRNGKey(3))
        data2 = _make_rdm_mock_data(jax.random.PRNGKey(4))
        create_likelihood = create_rdm_likelihood_factory_approx(rdm_conditioner)

        ll_fn1 = create_likelihood(data1)
        ll_fn2 = create_likelihood(data2)

        assert not jnp.allclose(jnp.sum(ll_fn1(rdm_params)), jnp.sum(ll_fn2(rdm_params)))

    def test_factory_returns_callable(self, rdm_conditioner):
        """The factory should return a create_likelihood function, which returns a likelihood_fun."""
        create_likelihood = create_rdm_likelihood_factory_approx(rdm_conditioner)
        assert callable(create_likelihood)

        data = _make_rdm_mock_data(jax.random.PRNGKey(5))
        ll_fn = create_likelihood(data)
        assert callable(ll_fn)

    def test_matches_analytical_shape(self, rdm_conditioner, rdm_params):
        """Approx likelihood should have the same output shape as the analytical one."""
        data = _make_rdm_mock_data(jax.random.PRNGKey(6))

        analytical_fn = create_rdm_two_accumulators_likelihood(data)
        create_likelihood = create_rdm_likelihood_factory_approx(rdm_conditioner)
        approx_fn = create_likelihood(data)

        assert analytical_fn(rdm_params).shape == approx_fn(rdm_params).shape


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
