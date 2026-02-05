import jax
import jax.numpy as jnp
import pytest
from confrdm_jax.simulators import gamma_pulse
from confrdm_jax.simulators import simulate_crdm_single_trial


class TestGammaPulse:
    def test_pulse_shape(self):
        t = jnp.linspace(0.01, 1.0, 100)
        pulse = gamma_pulse(t, amp=0.3, tau=0.1, a_shape=2.0)

        assert pulse.shape == t.shape
        assert jnp.all(jnp.isfinite(pulse))

    def test_pulse_decays_to_zero(self):
        t = jnp.linspace(0.01, 2.0, 1000)
        pulse = gamma_pulse(t, amp=0.3, tau=0.1, a_shape=2.0)

        # Pulse should decay toward zero at large t
        early_magnitude = jnp.abs(pulse[:100]).max()
        late_magnitude = jnp.abs(pulse[-100:]).max()
        assert late_magnitude < early_magnitude

    def test_pulse_amplitude_scaling(self):
        t = jnp.linspace(0.01, 1.0, 100)
        pulse1 = gamma_pulse(t, amp=0.3, tau=0.1, a_shape=2.0)
        pulse2 = gamma_pulse(t, amp=0.6, tau=0.1, a_shape=2.0)

        # Doubling amplitude should double the pulse
        assert jnp.allclose(pulse2, 2.0 * pulse1)

    def test_pulse_different_tau(self):
        t = jnp.linspace(0.01, 2.0, 500)
        pulse_fast = gamma_pulse(t, amp=0.3, tau=0.05, a_shape=2.0)
        pulse_slow = gamma_pulse(t, amp=0.3, tau=0.4, a_shape=2.0)

        # Different tau should produce different pulse shapes
        # The pulse with smaller tau should decay faster
        late_idx = -50  # Check values near the end
        assert jnp.abs(pulse_fast[late_idx]) < jnp.abs(pulse_slow[late_idx])


class TestSimulateCrdmSingleTrial:
    @pytest.mark.parametrize(("dt", "t_max"), ((0.001, 0.5), (0.01, 5.0), (0.1, 2.0)))
    def test_basic_simulation(self, dt, t_max, rng_key):
        t = jnp.expand_dims(jnp.arange(dt, t_max, dt), 0)
        mu = jnp.expand_dims(jnp.array([1.0, 4.0]), 1) * t
        sigma = jnp.expand_dims(jnp.array([1.0, 1.0]), 1)
        b = 1.0
        t0 = 0.3

        rt, resp = simulate_crdm_single_trial(mu, b, sigma, t0, dt, rng_key)

        assert jnp.all(jnp.bitwise_or(rt == -1, jnp.bitwise_and(rt > t0, rt <= t_max + t0)))
        assert jnp.all(jnp.bitwise_or(rt == -1, jnp.bitwise_or(resp == 0, resp == 1)))

    def test_no_boundary_crossing(self, rng_key):
        """Very high boundary with low drift should result in no crossing."""
        dt = 0.001
        t_max = 0.1  # Short time window
        t = jnp.expand_dims(jnp.arange(dt, t_max, dt), 0)
        mu = jnp.expand_dims(jnp.array([0.1, 0.1]), 1) * jnp.ones_like(t)  # Very low drift
        sigma = jnp.expand_dims(jnp.array([0.1, 0.1]), 1)
        b = 100.0  # Very high boundary
        t0 = 0.0

        rt, resp = simulate_crdm_single_trial(mu, b, sigma, t0, dt, rng_key)

        assert rt == -1.0
        assert resp == -1

    def test_fast_boundary_crossing(self, rng_key):
        """Very low boundary with high drift should result in fast crossing."""
        dt = 0.001
        t_max = 1.0
        t = jnp.expand_dims(jnp.arange(dt, t_max, dt), 0)
        mu = jnp.expand_dims(jnp.array([10.0, 10.0]), 1) * jnp.ones_like(t)  # High drift
        sigma = jnp.expand_dims(jnp.array([0.1, 0.1]), 1)  # Low noise
        b = 0.01  # Very low boundary
        t0 = 0.1

        rt, resp = simulate_crdm_single_trial(mu, b, sigma, t0, dt, rng_key)

        # Should cross quickly
        assert rt > t0
        assert rt < t0 + 0.1  # Should be very fast
        assert resp in [0, 1]

    def test_single_accumulator(self, rng_key):
        """Test with single accumulator (1D case)."""
        dt = 0.001
        t_max = 1.0
        t = jnp.arange(dt, t_max, dt)
        mu = jnp.expand_dims(jnp.array([5.0]) * t, 0)  # Single accumulator
        sigma = jnp.expand_dims(jnp.array([1.0]), 1)
        b = 1.0
        t0 = 0.2

        rt, resp = simulate_crdm_single_trial(mu, b, sigma, t0, dt, rng_key)

        assert rt > t0 or rt == -1.0
        assert resp == 0 or resp == -1  # Only one accumulator

    def test_asymmetric_drift_favors_faster(self, rng_key):
        """Accumulator with higher drift should win more often."""
        dt = 0.001
        t_max = 2.0
        t = jnp.expand_dims(jnp.arange(dt, t_max, dt), 0)

        # Accumulator 1 has much higher drift
        mu = jnp.vstack([
            jnp.ones_like(t) * 1.0,  # Slow
            jnp.ones_like(t) * 5.0,  # Fast
        ])
        sigma = jnp.expand_dims(jnp.array([1.0, 1.0]), 1)
        b = 1.0
        t0 = 0.0

        # Run many trials
        keys = jax.random.split(rng_key, 100)
        results = jax.vmap(
            lambda k: simulate_crdm_single_trial(mu, b, sigma, t0, dt, k)
        )(keys)

        responses = results[1]
        valid_responses = responses[responses != -1]

        # Accumulator 1 (faster) should win most of the time
        assert jnp.mean(valid_responses == 1) > 0.7

    def test_zero_noise(self, rng_key):
        """Test deterministic behavior with zero noise."""
        dt = 0.001
        t_max = 1.0
        t = jnp.expand_dims(jnp.arange(dt, t_max, dt), 0)
        mu = jnp.expand_dims(jnp.array([2.0, 4.0]), 1) * jnp.ones_like(t)
        sigma = jnp.expand_dims(jnp.array([0.0, 0.0]), 1)  # No noise
        b = 1.0
        t0 = 0.1

        # With zero noise, faster accumulator should always win
        keys = jax.random.split(rng_key, 10)
        results = jax.vmap(
            lambda k: simulate_crdm_single_trial(mu, b, sigma, t0, dt, k)
        )(keys)

        responses = results[1]
        # Accumulator 1 should always win (higher drift)
        assert jnp.all(responses == 1)
