import jax.numpy as jnp
from confrdm_jax.simulators import create_wald_prior_uniform
from confrdm_jax.simulators import sample_conditional_wald


class TestWaldModel:
    def test_sample_conditional_informed_prior(self, rng_key, batch_shape, prior_wald):
        data, context = sample_conditional_wald(rng_key, batch_shape, prior_wald)

        assert data.shape == batch_shape + (1,)
        assert context.shape == (batch_shape[0], 1, 3)
        assert jnp.all(jnp.isfinite(data))
        assert jnp.all(jnp.isfinite(context))

    def test_sample_conditional_uniform_prior(self, rng_key, batch_shape, prior_wald_uniform):
        data, context = sample_conditional_wald(rng_key, batch_shape, prior_wald_uniform)

        assert data.shape == batch_shape + (1,)
        assert context.shape == (batch_shape[0], 1, 3)
        assert jnp.all(data > 0)  # Wald samples should be positive
        assert jnp.all(jnp.isfinite(data))

    def test_prior_uniform_bounds(self, rng_key, prior_wald_uniform):
        samples = prior_wald_uniform.sample(seed=rng_key, sample_shape=(1000,))

        # Check all samples are within default bounds
        v, s, b = samples[0], samples[1], samples[2]
        assert jnp.all(v >= 0.0) and jnp.all(v <= 8.0)
        assert jnp.all(s >= 0.0) and jnp.all(s <= 2.0)
        assert jnp.all(b >= 0.0) and jnp.all(b <= 2.0)

    def test_prior_informed_positive(self, rng_key, prior_wald):
        samples = prior_wald.sample(seed=rng_key, sample_shape=(1000,))

        # Informed prior uses truncated normals, should be positive
        for i in range(3):
            assert jnp.all(samples[i] >= 0.0)

    def test_custom_prior_parameters(self, rng_key):
        """Test that custom prior parameters are respected."""
        prior = create_wald_prior_uniform(v_min=1.0, v_max=2.0)
        samples = prior.sample(seed=rng_key, sample_shape=(1000,))

        v = samples[0]
        assert jnp.all(v >= 1.0) and jnp.all(v <= 2.0)
