import jax
import jax.numpy as jnp
from confrdm_jax.simulators import sample_conditional_crdm_single
from confrdm_jax.simulators import simulate_crdm_single_batch
from confrdm_jax.simulators import simulate_crdm_single_dataset


class TestCrdmSingleModel:
    def test_simulate_dataset(self, rng_key, sim_config, crdm_single_params):
        keys = jax.random.split(rng_key, sim_config["num_obs"])

        rt = simulate_crdm_single_dataset(
            keys,
            crdm_single_params["v_c"],
            crdm_single_params["amp"],
            crdm_single_params["tau"],
            crdm_single_params["s"],
            crdm_single_params["b"],
            crdm_single_params["t0"],
            sim_config["dt"],
            sim_config["t_max"],
        )

        assert rt.shape[0] == sim_config["num_obs"]
        assert jnp.all(rt > crdm_single_params["t0"])

    def test_simulate_batch(self, rng_key, sim_config, crdm_single_params_batch):
        batch_size = sim_config["batch_size"]
        num_obs = sim_config["num_obs"]
        keys = jax.random.split(rng_key, (batch_size, num_obs))

        rt = simulate_crdm_single_batch(
            keys,
            crdm_single_params_batch["v_c"],
            crdm_single_params_batch["amp"],
            crdm_single_params_batch["tau"],
            crdm_single_params_batch["s"],
            crdm_single_params_batch["b"],
            crdm_single_params_batch["t0"],
            sim_config["dt"],
            sim_config["t_max"],
        )

        assert rt.shape == (batch_size, num_obs)
        # Make sure that RT is above t0
        assert jnp.all(rt > crdm_single_params_batch["t0"][..., None])
        # Check that RT and responses are not the same (i.e., random number generation works)
        assert jnp.all(jnp.isfinite(jnp.var(rt, axis=1)))

    def test_batch_reproducibility(self, rng_key, sim_config, crdm_single_params_batch):
        batch_size = sim_config["batch_size"]
        num_obs = sim_config["num_obs"]
        keys = jax.random.split(rng_key, (batch_size, num_obs))

        rt = simulate_crdm_single_batch(
            keys,
            crdm_single_params_batch["v_c"],
            crdm_single_params_batch["amp"],
            crdm_single_params_batch["tau"],
            crdm_single_params_batch["s"],
            crdm_single_params_batch["b"],
            crdm_single_params_batch["t0"],
            sim_config["dt"],
            sim_config["t_max"],
        )

        rt_new = simulate_crdm_single_batch(
            keys,
            crdm_single_params_batch["v_c"],
            crdm_single_params_batch["amp"],
            crdm_single_params_batch["tau"],
            crdm_single_params_batch["s"],
            crdm_single_params_batch["b"],
            crdm_single_params_batch["t0"],
            sim_config["dt"],
            sim_config["t_max"],
        )

        assert jnp.all(rt == rt_new)

    def test_sample_conditional(self, rng_key, batch_shape, sim_config, prior_crdm_single):
        data, context = sample_conditional_crdm_single(
            rng_key, batch_shape, prior_crdm_single, sim_config["dt"], sim_config["t_max"]
        )

        assert data.shape == batch_shape + (1,)
        assert context.shape == (batch_shape[0], 1, 5)
        assert jnp.all(jnp.isfinite(data))
        assert jnp.all(jnp.isfinite(context))

    def test_prior_uniform_bounds(self, rng_key, prior_crdm_single_uniform):
        samples = prior_crdm_single_uniform.sample(seed=rng_key, sample_shape=(1000,))

        # Check default bounds
        assert jnp.all(samples[0] >= 0.0) and jnp.all(samples[0] <= 8.0)  # v_c
        assert jnp.all(samples[1] >= 0.0) and jnp.all(samples[1] <= 0.5)  # amp
        assert jnp.all(samples[2] >= 0.0) and jnp.all(samples[2] <= 0.5)  # tau
        assert jnp.all(samples[3] >= 0.0) and jnp.all(samples[3] <= 2.0)  # s
        assert jnp.all(samples[4] >= 0.0) and jnp.all(samples[4] <= 2.0)  # b

    def test_prior_informed_positive(self, rng_key, prior_crdm_single):
        samples = prior_crdm_single.sample(seed=rng_key, sample_shape=(1000,))

        for i in range(5):
            assert jnp.all(samples[i] >= 0.0)
