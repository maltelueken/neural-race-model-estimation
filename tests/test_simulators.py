
import jax
import jax.numpy as jnp
import pytest
from confrdm_jax.simulators import create_crdm_single_prior_informed
from confrdm_jax.simulators import create_wald_prior_informed
from confrdm_jax.simulators import sample_conditional_crdm_single
from confrdm_jax.simulators import sample_conditional_wald
from confrdm_jax.simulators import simulate_crdm_batch
from confrdm_jax.simulators import simulate_crdm_dataset
from confrdm_jax.simulators import simulate_crdm_single_batch
from confrdm_jax.simulators import simulate_crdm_single_dataset
from confrdm_jax.simulators import simulate_crdm_single_trial


# --- Fixtures ---


@pytest.fixture
def rng_key():
    return jax.random.key(123)


@pytest.fixture
def sim_config():
    return {
        "dt": 0.001,
        "t_max": 5.0,
        "num_obs": 100,
        "batch_size": 2,
    }


@pytest.fixture
def crdm_params():
    return {
        "v_c_intercept": 1.0,
        "v_c_slope": 4.0,
        "amp": 0.3,
        "tau": 0.15,
        "s_true": 1.0,
        "b": 1.0,
        "t0": 0.3,
    }


@pytest.fixture
def crdm_params_batch(crdm_params, sim_config):
    batch_size = sim_config["batch_size"]
    return {k: jnp.array([v] * batch_size) for k, v in crdm_params.items()}


@pytest.fixture
def crdm_single_params():
    return {
        "v_c": 4.0,
        "amp": 0.3,
        "tau": 0.15,
        "s": 1.0,
        "b": 1.0,
        "t0": 0.3,
    }


@pytest.fixture
def crdm_single_params_batch(crdm_single_params, sim_config):
    batch_size = sim_config["batch_size"]
    return {k: jnp.array([v] * batch_size) for k, v in crdm_single_params.items()}


@pytest.fixture
def batch_shape():
    return (10, 20)


@pytest.fixture
def prior_wald():
    return create_wald_prior_informed()


@pytest.fixture
def prior_crdm_single():
    return create_crdm_single_prior_informed()


# --- Tests ---


@pytest.mark.parametrize(("dt", "t_max"), ((0.001, 0.5), (0.01, 5.0), (0.1, 2.0)))
def test_simulate_crdm_single_trial(dt, t_max, rng_key):
    t = jnp.expand_dims(jnp.arange(dt, t_max, dt), 0)
    mu = jnp.expand_dims(jnp.array([1.0, 4.0]), 1) * t
    sigma = jnp.expand_dims(jnp.array([1.0, 1.0]), 1)
    b = 1.0
    t0 = 0.3

    rt, resp = simulate_crdm_single_trial(mu, b, sigma, t0, dt, rng_key)

    assert jnp.all(jnp.bitwise_or(rt == -1, jnp.bitwise_and(rt > t0, rt <= t_max + t0)))
    assert jnp.all(jnp.bitwise_or(rt == -1, jnp.bitwise_or(resp == 0, resp == 1)))


def test_simulate_crdm_dataset(rng_key, sim_config, crdm_params):
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


def test_simulate_crdm_batch(rng_key, sim_config, crdm_params_batch):
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

    # Check that reproducibility works
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

    rt_new = data_new[..., 0]
    resp_new = data_new[..., 1]

    assert jnp.all(rt == rt_new)
    assert jnp.all(resp == resp_new)


def test_simulate_crdm_single_dataset(rng_key, sim_config, crdm_single_params):
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


def test_simulate_crdm_single_batch(rng_key, sim_config, crdm_single_params_batch):
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

    # Check that reproducibility works
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


def test_sample_conditional_wald(rng_key, batch_shape, prior_wald):
    data, context = sample_conditional_wald(rng_key, batch_shape, prior_wald)

    assert data.shape == batch_shape + (1,)
    assert context.shape == (batch_shape[0], 1, 3)
    assert jnp.all(jnp.isfinite(data))
    assert jnp.all(jnp.isfinite(context))


def test_sample_conditional_crdm_single(rng_key, batch_shape, sim_config, prior_crdm_single):
    data, context = sample_conditional_crdm_single(
        rng_key, batch_shape, prior_crdm_single, sim_config["dt"], sim_config["t_max"]
    )

    assert data.shape == batch_shape + (1,)
    assert context.shape == (batch_shape[0], 1, 5)
    assert jnp.all(jnp.isfinite(data))
    assert jnp.all(jnp.isfinite(context))
