import jax
import jax.numpy as jnp
import pytest
from confrdm_jax.simulators import create_crdm_prior_informed
from confrdm_jax.simulators import create_crdm_single_prior_informed
from confrdm_jax.simulators import create_crdm_single_prior_uniform
from confrdm_jax.simulators import create_rdm_prior_informed
from confrdm_jax.simulators import create_wald_prior_informed
from confrdm_jax.simulators import create_wald_prior_uniform


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
def batch_shape():
    return (10, 20)


# --- Wald Fixtures ---


@pytest.fixture
def prior_wald():
    return create_wald_prior_informed()


@pytest.fixture
def prior_wald_uniform():
    return create_wald_prior_uniform()


# --- RDM Fixtures ---


@pytest.fixture
def prior_rdm():
    return create_rdm_prior_informed()


# --- CRDM Fixtures ---


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
def prior_crdm():
    return create_crdm_prior_informed()


# --- CRDM Single Fixtures ---


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
def prior_crdm_single():
    return create_crdm_single_prior_informed()


@pytest.fixture
def prior_crdm_single_uniform():
    return create_crdm_single_prior_uniform()
