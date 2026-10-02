"""Shared fixtures.

64-bit precision is enabled for the whole session, because the race log-densities under test
are precision-sensitive and the tolerances here assume it.
"""

import jax
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402  (must follow the x64 switch)
from flax import nnx  # noqa: E402

from confrdm_jax.priors import (  # noqa: E402
    create_crdm_prior_informed,
    create_crdm_single_prior_uniform,
    create_rdm_prior_informed,
    create_wald_prior_uniform,
)


@pytest.fixture(scope="session")
def rdm_prior():
    return create_rdm_prior_informed()


@pytest.fixture(scope="session")
def crdm_prior():
    return create_crdm_prior_informed()


@pytest.fixture(scope="session")
def wald_training_prior():
    return create_wald_prior_uniform()


@pytest.fixture(scope="session")
def crdm_training_prior():
    return create_crdm_single_prior_uniform()


@pytest.fixture(scope="session")
def rdm_theta():
    """A plausible RDM parameter vector in log space."""
    return jnp.log(jnp.array([1.0, 1.5, 1.2, 1.2, 0.3]))


@pytest.fixture(scope="session")
def crdm_theta():
    """A plausible CRDM parameter vector in log space."""
    return jnp.log(jnp.array([1.0, 4.0, 0.3, 0.1, 1.0, 0.8, 0.3]))


def _conditioner(num_in, seed):
    from eamax.flows import make_mlp_conditioner

    return make_mlp_conditioner(
        rngs=nnx.Rngs(default=jax.random.key(seed)), num_in=num_in, num_mid=16, num_bins=4
    )


@pytest.fixture(scope="session")
def wald_conditioner():
    """An *untrained* 3-input conditioner.

    Untrained is enough for every test here: they check that the flow is wired into the race
    correctly — which accumulator it scores, what happens to censored and masked trials —
    not that its density is accurate. Accuracy is what
    ``scripts/compare_neural_densities.py`` measures against the Volterra solver.
    """
    return _conditioner(3, 0)


@pytest.fixture(scope="session")
def crdm_conditioner():
    """An untrained 5-input conditioner; see :func:`wald_conditioner`."""
    return _conditioner(5, 1)
