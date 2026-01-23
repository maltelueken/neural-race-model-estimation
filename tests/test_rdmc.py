
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


@pytest.mark.parametrize(("dt", "t_max"), ((0.001, 0.5), (0.01, 5.0), (0.1, 2.0)))
def test_simulate_crdm_single_trial(dt, t_max):
    t = jnp.expand_dims(jnp.arange(dt, t_max, dt), 0)
    mu = jnp.expand_dims(jnp.array([1.0, 4.0]), 1) * t
    sigma = jnp.expand_dims(jnp.array([1.0, 1.0]), 1)
    b = 1.0
    t0 = 0.3

    key = jax.random.key(123)

    rt, resp = simulate_crdm_single_trial(mu, b, sigma, t0, dt, key)

    assert jnp.all(jnp.bitwise_or(rt == -1, jnp.bitwise_and(rt > t0, rt <= t_max + t0)))
    assert jnp.all(jnp.bitwise_or(rt == -1, jnp.bitwise_or(resp == 0, resp == 1)))


def test_simulate_crdm_dataset():
    v_c_intercept = 1.0
    v_c_slope = 4.0
    amp = 0.3
    tau = 0.15
    s_true = 1.0
    b = 1.0
    t0 = 0.3
    num_obs = 100
    dt = 0.001
    t_max = 5.0

    key = jax.random.key(123)
    keys = jax.random.split(key, num_obs)

    data = simulate_crdm_dataset(
        keys, v_c_intercept, v_c_slope, amp, tau, s_true, b, t0, dt, t_max,
    )

    rt = data[:, 0]
    resp = data[:, 1]

    assert rt.shape[0] == num_obs
    assert resp.shape[0] == num_obs
    assert jnp.all(rt > t0)
    assert jnp.all(jnp.bitwise_or(resp == 0, resp == 1))
    assert jnp.mean(resp) < 1.0


def test_simulate_crdm_batch():
    v_c_intercept = jnp.array([1.0, 1.0])
    v_c_slope = jnp.array([4.0, 4.0])
    amp = jnp.array([0.3, 0.3])
    tau = jnp.array([0.15, 0.15])
    s_true = jnp.array([1.0, 1.0])
    b = jnp.array([1.0, 1.0])
    t0 = jnp.array([0.3, 0.3])
    dt = 0.001
    t_max = 5.0
    num_obs = 100
    batch_size = 2

    key = jax.random.key(123)
    keys = jax.random.split(key, (batch_size, num_obs))

    data = simulate_crdm_batch(
        keys, v_c_intercept, v_c_slope, amp, tau, s_true, b, t0, dt, t_max,
    )

    rt = data[..., 0]
    resp = data[..., 1]

    assert rt.shape == (batch_size, num_obs)
    assert resp.shape == (batch_size, num_obs)
    # Make sure that RT is above t0
    assert jnp.all(rt > t0[..., None])
    assert jnp.all(jnp.bitwise_or(resp == 0, resp == 1))
    # Check that RT and responses are not the same (i.e., random number generation works)
    assert jnp.all(jnp.isfinite(jnp.var(rt, axis=1)))
    assert jnp.all(jnp.isfinite(jnp.var(resp, axis=1)))

    # Check that reproducibility works
    data_new = simulate_crdm_batch(
        keys, v_c_intercept, v_c_slope, amp, tau, s_true, b, t0, dt, t_max,
    )

    rt_new = data_new[..., 0]
    resp_new = data_new[..., 1]

    assert jnp.all(rt == rt_new)
    assert jnp.all(resp == resp_new)


def test_simulate_crdm_single_dataset():
    v_c = 4.0
    amp = 0.3
    tau = 0.15
    s = 1.0
    b = 1.0
    t0 = 0.3
    num_obs = 100
    dt = 0.001
    t_max = 5.0

    key = jax.random.key(123)
    keys = jax.random.split(key, num_obs)

    rt = simulate_crdm_single_dataset(keys, v_c, amp, tau, s, b, t0, dt, t_max)

    assert rt.shape[0] == num_obs
    assert jnp.all(rt > t0)


def test_simulate_crdm_single_batch():
    v_c = jnp.array([4.0, 4.0])
    amp = jnp.array([0.3, 0.3])
    tau = jnp.array([0.15, 0.15])
    s = jnp.array([1.0, 1.0])
    b = jnp.array([1.0, 1.0])
    t0 = jnp.array([0.3, 0.3])
    dt = 0.001
    t_max = 5.0
    num_obs = 100
    batch_size = 2

    key = jax.random.key(123)
    keys = jax.random.split(key, (batch_size, num_obs))

    rt = simulate_crdm_single_batch(
        keys, v_c, amp, tau, s, b, t0, dt, t_max,
    )

    assert rt.shape == (batch_size, num_obs)
    # Make sure that RT is above t0
    assert jnp.all(rt > t0[..., None])
    # Check that RT and responses are not the same (i.e., random number generation works)
    assert jnp.all(jnp.isfinite(jnp.var(rt, axis=1)))

    # Check that reproducibility works
    rt_new = simulate_crdm_single_batch(
        keys, v_c, amp, tau, s, b, t0, dt, t_max,
    )

    assert jnp.all(rt == rt_new)


@pytest.fixture
def prior_wald():
    return create_wald_prior_informed()


def test_sample_conditional_wald(prior_wald):
    key = jax.random.key(123)
    batch_shape = (10, 20)
    data, context = sample_conditional_wald(key, batch_shape, prior_wald)

    assert data.shape == batch_shape + (1,)
    assert context.shape == (batch_shape[0], 1, 3)
    assert jnp.all(jnp.isfinite(data))
    assert jnp.all(jnp.isfinite(context))


@pytest.fixture
def prior_crdm_single():
    return create_crdm_single_prior_informed()


def test_sample_conditional_crdm_single(prior_crdm_single):
    key = jax.random.key(123)
    batch_shape = (10, 20)
    dt = 0.001
    t_max = 5.0
    data, context = sample_conditional_crdm_single(key, batch_shape, prior_crdm_single, dt, t_max)

    assert data.shape == batch_shape + (1,)
    assert context.shape == (batch_shape[0], 1, 5)
    assert jnp.all(jnp.isfinite(data))
    assert jnp.all(jnp.isfinite(context))
