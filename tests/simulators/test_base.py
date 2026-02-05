import jax.numpy as jnp
from confrdm_jax.simulators import TruncatedNormal


class TestTruncatedNormal:
    def test_samples_within_bounds(self, rng_key):
        dist = TruncatedNormal(loc=0.0, scale=1.0, lower=-1.0, upper=1.0)
        samples = dist.sample(seed=rng_key, sample_shape=(1000,))

        assert jnp.all(samples >= -1.0)
        assert jnp.all(samples <= 1.0)

    def test_samples_positive_only(self, rng_key):
        dist = TruncatedNormal(loc=1.0, scale=0.5, lower=0.0, upper=jnp.inf)
        samples = dist.sample(seed=rng_key, sample_shape=(1000,))

        assert jnp.all(samples >= 0.0)
        assert jnp.all(jnp.isfinite(samples))

    def test_log_prob_within_bounds(self):
        dist = TruncatedNormal(loc=0.0, scale=1.0, lower=-1.0, upper=1.0)

        lp_inside = dist.log_prob(0.0)
        lp_at_lower = dist.log_prob(-1.0)
        lp_at_upper = dist.log_prob(1.0)

        assert jnp.isfinite(lp_inside)
        assert jnp.isfinite(lp_at_lower)
        assert jnp.isfinite(lp_at_upper)

    def test_log_prob_outside_bounds(self):
        dist = TruncatedNormal(loc=0.0, scale=1.0, lower=-1.0, upper=1.0)

        lp_below = dist.log_prob(-2.0)
        lp_above = dist.log_prob(2.0)

        assert lp_below == -jnp.inf
        assert lp_above == -jnp.inf

    def test_mode_inside_bounds(self):
        dist = TruncatedNormal(loc=0.5, scale=1.0, lower=0.0, upper=1.0)
        assert dist.mode() == 0.5

    def test_mode_below_lower_bound(self):
        dist = TruncatedNormal(loc=-1.0, scale=1.0, lower=0.0, upper=2.0)
        assert dist.mode() == 0.0

    def test_mode_above_upper_bound(self):
        dist = TruncatedNormal(loc=3.0, scale=1.0, lower=0.0, upper=2.0)
        assert dist.mode() == 2.0

    def test_batch_shape(self):
        dist = TruncatedNormal(
            loc=jnp.array([0.0, 1.0]),
            scale=jnp.array([1.0, 0.5]),
            lower=0.0,
            upper=jnp.inf,
        )
        assert dist.batch_shape == (2,)

    def test_event_shape(self):
        dist = TruncatedNormal(loc=0.0, scale=1.0, lower=0.0, upper=1.0)
        assert dist.event_shape == ()
