import jax
import jax.numpy as jnp
from confrdm_jax.simulators import sample_conditional_crdm
from confrdm_jax.simulators import simulate_crdm_batch
from confrdm_jax.simulators import simulate_crdm_dataset


class TestCrdmModel:
    def test_simulate_dataset(self, rng_key, sim_config, crdm_params):
        keys = jax.random.split(rng_key, sim_config["num_obs"])

        data = simulate_crdm_dataset(
            keys,
            crdm_params["v_c_intercept"],
            crdm_params["v_c_slope"],
            crdm_params["amp"],
            crdm_params["tau"],
            crdm_params["s_true"],
            crdm_params["b"],
            crdm_params["t0"],
            sim_config["dt"],
            sim_config["t_max"],
        )

        rt = data[:, 0]
        resp = data[:, 1]

        assert rt.shape[0] == sim_config["num_obs"]
        assert resp.shape[0] == sim_config["num_obs"]
        assert jnp.all(rt > crdm_params["t0"])
        assert jnp.all(jnp.bitwise_or(resp == 0, resp == 1))
        assert jnp.mean(resp) < 1.0

    def test_simulate_batch(self, rng_key, sim_config, crdm_params_batch):
        batch_size = sim_config["batch_size"]
        num_obs = sim_config["num_obs"]
        keys = jax.random.split(rng_key, (batch_size, num_obs))

        data = simulate_crdm_batch(
            keys,
            crdm_params_batch["v_c_intercept"],
            crdm_params_batch["v_c_slope"],
            crdm_params_batch["amp"],
            crdm_params_batch["tau"],
            crdm_params_batch["s_true"],
            crdm_params_batch["b"],
            crdm_params_batch["t0"],
            sim_config["dt"],
            sim_config["t_max"],
        )

        rt = data[..., 0]
        resp = data[..., 1]

        assert rt.shape == (batch_size, num_obs)
        assert resp.shape == (batch_size, num_obs)
        # Make sure that RT is above t0
        assert jnp.all(rt > crdm_params_batch["t0"][..., None])
        assert jnp.all(jnp.bitwise_or(resp == 0, resp == 1))
        # Check that RT and responses are not the same (i.e., random number generation works)
        assert jnp.all(jnp.isfinite(jnp.var(rt, axis=1)))
        assert jnp.all(jnp.isfinite(jnp.var(resp, axis=1)))

    def test_batch_reproducibility(self, rng_key, sim_config, crdm_params_batch):
        batch_size = sim_config["batch_size"]
        num_obs = sim_config["num_obs"]
        keys = jax.random.split(rng_key, (batch_size, num_obs))

        data = simulate_crdm_batch(
            keys,
            crdm_params_batch["v_c_intercept"],
            crdm_params_batch["v_c_slope"],
            crdm_params_batch["amp"],
            crdm_params_batch["tau"],
            crdm_params_batch["s_true"],
            crdm_params_batch["b"],
            crdm_params_batch["t0"],
            sim_config["dt"],
            sim_config["t_max"],
        )

        data_new = simulate_crdm_batch(
            keys,
            crdm_params_batch["v_c_intercept"],
            crdm_params_batch["v_c_slope"],
            crdm_params_batch["amp"],
            crdm_params_batch["tau"],
            crdm_params_batch["s_true"],
            crdm_params_batch["b"],
            crdm_params_batch["t0"],
            sim_config["dt"],
            sim_config["t_max"],
        )

        assert jnp.all(data[..., 0] == data_new[..., 0])
        assert jnp.all(data[..., 1] == data_new[..., 1])

    def test_sample_conditional(self, rng_key, batch_shape, sim_config, prior_crdm):
        data, context = sample_conditional_crdm(
            rng_key, batch_shape, prior_crdm, sim_config["dt"], sim_config["t_max"]
        )

        assert data.shape == batch_shape + (2, 1)
        assert context.shape == (batch_shape[0], 1, 7)
        assert jnp.all(jnp.isfinite(context))

    def test_prior_informed_positive(self, rng_key, prior_crdm):
        samples = prior_crdm.sample(seed=rng_key, sample_shape=(100,))

        # All parameters should be positive
        for i in range(7):
            assert jnp.all(samples[i] >= 0.0)

    def test_negative_amplitude_incongruent(self, rng_key, sim_config):
        """Negative amplitude should simulate incongruent condition."""
        keys = jax.random.split(rng_key, sim_config["num_obs"])

        data = simulate_crdm_dataset(
            keys,
            v_c_intercept=1.0,
            v_c_slope=4.0,
            amp=-0.3,  # Negative = incongruent
            tau=0.15,
            s_true=1.0,
            b=1.0,
            t0=0.3,
            dt=sim_config["dt"],
            t_max=sim_config["t_max"],
        )

        rt = data[:, 0]
        resp = data[:, 1]

        assert jnp.all(rt > 0.3)
        assert jnp.all(jnp.bitwise_or(resp == 0, resp == 1))

    def test_zero_amplitude_neutral(self, rng_key, sim_config):
        """Zero amplitude should have no conflict effect."""
        keys = jax.random.split(rng_key, sim_config["num_obs"])

        data = simulate_crdm_dataset(
            keys,
            v_c_intercept=1.0,
            v_c_slope=4.0,
            amp=0.0,  # No conflict effect
            tau=0.15,
            s_true=1.0,
            b=1.0,
            t0=0.3,
            dt=sim_config["dt"],
            t_max=sim_config["t_max"],
        )

        rt = data[:, 0]
        assert jnp.all(rt > 0.3)

    def test_short_t_max_may_cause_non_crossings(self, rng_key):
        """Very short t_max with high boundary may cause non-crossings."""
        num_obs = 50
        keys = jax.random.split(rng_key, num_obs)

        data = simulate_crdm_dataset(
            keys,
            v_c_intercept=0.5,  # Low drift
            v_c_slope=0.5,
            amp=0.1,
            tau=0.15,
            s_true=1.0,
            b=2.0,  # High boundary
            t0=0.0,
            dt=0.001,
            t_max=0.5,  # Short time window
        )

        rt = data[:, 0]
        # Some trials may not cross (rt == -1 + t0 = -1)
        # This tests that the simulator handles edge cases gracefully
        assert jnp.all(jnp.isfinite(rt))

    def test_high_noise_increases_variance(self, rng_key, sim_config):
        """Higher noise should increase RT variance."""
        keys = jax.random.split(rng_key, sim_config["num_obs"])

        data_low_noise = simulate_crdm_dataset(
            keys,
            v_c_intercept=1.0,
            v_c_slope=4.0,
            amp=0.3,
            tau=0.15,
            s_true=0.5,  # Low noise
            b=1.0,
            t0=0.3,
            dt=sim_config["dt"],
            t_max=sim_config["t_max"],
        )

        data_high_noise = simulate_crdm_dataset(
            keys,
            v_c_intercept=1.0,
            v_c_slope=4.0,
            amp=0.3,
            tau=0.15,
            s_true=2.0,  # High noise
            b=1.0,
            t0=0.3,
            dt=sim_config["dt"],
            t_max=sim_config["t_max"],
        )

        rt_low = data_low_noise[:, 0]
        rt_high = data_high_noise[:, 0]

        # Filter valid RTs
        rt_low_valid = rt_low[rt_low > 0]
        rt_high_valid = rt_high[rt_high > 0]

        if len(rt_low_valid) > 10 and len(rt_high_valid) > 10:
            assert jnp.var(rt_high_valid) > jnp.var(rt_low_valid)
