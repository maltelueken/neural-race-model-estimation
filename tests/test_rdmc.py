
import jax
import jax.numpy as jnp
import pytest

from confrdm.rdmc import rdmc_experiment_simple_jax, rdmc_experiment_simple_batched, RDMCSimulator, rdmc_single_trial


@pytest.mark.parametrize("t_max", (50, 500, 5000))
def test_rdmc_single_trial(t_max):
    mu = jnp.tile(jnp.array([0.5, 0.5]), (t_max, 1)).T
    sigma = jnp.array([1.0, 1.0])
    b = 70
    t0 = 300

    key = jax.random.key(123)

    rt, resp = rdmc_single_trial(mu, b, sigma, t0, key)

    assert jnp.all(jnp.bitwise_or(rt == -1, jnp.bitwise_and(rt > t0, rt <= t_max + t0)))
    assert jnp.all(jnp.bitwise_or(rt == -1, jnp.bitwise_or(resp == 0, resp == 1)))


def test_rdmc_experiment_simple_jax():
    v_c_intercept=0.05
    v_c_slope=0.5
    amp=20.0
    tau=80.0
    s_true=4.0
    s_false=4.0
    b=70.0
    t0=300.0
    a_shape=2.0
    num_obs = 100
    t_max = 500

    key = jax.random.key(123)
    keys = jax.random.split(key, num_obs)

    rt, resp = rdmc_experiment_simple_jax(
        keys, v_c_intercept, v_c_slope, amp, tau, s_true, s_false, b, t0, a_shape, t_max
    )

    assert rt.shape[0] == num_obs
    assert resp.shape[0] == num_obs
    assert jnp.all(rt > t0)
    assert jnp.all(jnp.bitwise_or(resp == 0, resp == 1))
    assert jnp.mean(resp) < 1.0


def test_rdmc_experiment_simple_batched():
    v_c_intercept = jnp.array([0.05, 0.05])
    v_c_slope = jnp.array([0.5, 0.5])
    amp = jnp.array([20.0, 20.0])
    tau = jnp.array([80.0, 80.0])
    s_true = jnp.array([4.0, 4.0])
    s_false = jnp.array([4.0, 4.0])
    b = jnp.array([70.0, 70.0,])
    t0 = jnp.array([300.0, 300.0])
    a_shape = jnp.array([2.0, 2.0])
    t_max = 500
    num_obs = 100
    batch_size = 2

    key = jax.random.key(123)
    keys = jax.random.split(key, (batch_size, num_obs))

    rt, resp = rdmc_experiment_simple_batched(
        keys, v_c_intercept, v_c_slope, amp, tau, s_true, s_false, b, t0, a_shape, t_max
    )

    assert rt.shape == (batch_size, num_obs)
    assert resp.shape == (batch_size, num_obs)
    # Make sure that RT is above t0
    assert jnp.all(rt > t0[..., None])
    assert jnp.all(jnp.bitwise_or(resp == 0, resp == 1))
    # Check that RT and responses are not the same (i.e., random number generation works)
    assert jnp.all(jnp.isfinite(jnp.var(rt, axis=1)))
    assert jnp.all(jnp.isfinite(jnp.var(resp, axis=1)))

    # Check that reproducibility works
    rt_new, resp_new = rdmc_experiment_simple_batched(
        keys, v_c_intercept, v_c_slope, amp, tau, s_true, s_false, b, t0, a_shape, t_max
    )
    assert jnp.all(rt == rt_new)
    assert jnp.all(resp == resp_new)


class TestRDMCSimulator:
    batch_size = 10

    @pytest.fixture
    def simulator(self, start_seed=2025):
        def num_obs_fun(batch_shape, key):
            return jax.random.randint(key, (), 100, 1000)
        
        return RDMCSimulator(num_obs_fun=num_obs_fun, start_seed=start_seed)

    def test_rdmc_simulator(self, simulator):
        data = simulator.sample((self.batch_size,))
        
        num_obs = data.pop("num_obs")

        assert num_obs.shape == ()

        for val in data.values():
            assert val.shape[0] == self.batch_size

        # Check that repeated sampling gives different results
        data_new = simulator.sample((self.batch_size,))

        assert num_obs != data_new["num_obs"]
        assert jnp.all(jnp.mean(data["x"], axis=1) != jnp.mean(data_new["x"], axis=1))

    def test_rdmc_simulator_kwargs(self, simulator):
        key = jax.random.key(2025)
        num_obs = 100
        data = simulator.sample((self.batch_size,), key=key, num_obs=num_obs)

        assert data["num_obs"] == num_obs

        a_shape = 10
        data_new = simulator.sample((self.batch_size,), key=key, num_obs=num_obs, a_shape=a_shape)
        # Only compare mean RT because noise is identical leading to identical responses
        assert jnp.all(jnp.mean(data["x"][..., 0]) != jnp.mean(data_new["x"][..., 0]))
