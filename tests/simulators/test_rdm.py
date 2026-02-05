import jax
import jax.numpy as jnp
from confrdm_jax.simulators import sample_conditional_rdm
from confrdm_jax.simulators import simulate_rdm


class TestRdmModel:
    def test_simulate_basic(self, rng_key):
        data = simulate_rdm(
            v_intercept=1.0,
            v_slope=1.5,
            s_true=1.2,
            b=1.0,
            t0=0.3,
            batch_shape=(100,),
            key=rng_key,
        )

        rt = data[:, 0]
        resp = data[:, 1]

        assert data.shape == (100, 2)
        assert jnp.all(rt > 0.3)  # RT > t0
        assert jnp.all(jnp.bitwise_or(resp == 0, resp == 1))

    def test_simulate_slope_affects_accuracy(self, rng_key):
        """Higher v_slope should increase accuracy (more resp=1)."""
        data_low_slope = simulate_rdm(
            v_intercept=1.0,
            v_slope=0.5,  # Low slope
            s_true=1.0,
            b=1.0,
            t0=0.3,
            batch_shape=(500,),
            key=rng_key,
        )

        key2 = jax.random.split(rng_key)[0]
        data_high_slope = simulate_rdm(
            v_intercept=1.0,
            v_slope=3.0,  # High slope
            s_true=1.0,
            b=1.0,
            t0=0.3,
            batch_shape=(500,),
            key=key2,
        )

        # Higher slope = faster "correct" accumulator = more resp=1
        assert jnp.mean(data_high_slope[:, 1]) > jnp.mean(data_low_slope[:, 1])

    def test_sample_conditional(self, rng_key, batch_shape, prior_rdm):
        data, context = sample_conditional_rdm(rng_key, batch_shape, prior_rdm)

        assert data.shape == batch_shape + (2,)
        assert context.shape == (batch_shape[0], 1, 5)
        assert jnp.all(jnp.isfinite(data))
        assert jnp.all(jnp.isfinite(context))

    def test_prior_informed_positive(self, rng_key, prior_rdm):
        samples = prior_rdm.sample(seed=rng_key, sample_shape=(1000,))

        # All parameters should be positive
        for i in range(5):
            assert jnp.all(samples[i] >= 0.0)

    def test_reproducibility(self, rng_key):
        """Same key should produce same results."""
        data1 = simulate_rdm(
            v_intercept=1.0,
            v_slope=1.5,
            s_true=1.2,
            b=1.0,
            t0=0.3,
            batch_shape=(50,),
            key=rng_key,
        )

        data2 = simulate_rdm(
            v_intercept=1.0,
            v_slope=1.5,
            s_true=1.2,
            b=1.0,
            t0=0.3,
            batch_shape=(50,),
            key=rng_key,
        )

        assert jnp.allclose(data1, data2)
