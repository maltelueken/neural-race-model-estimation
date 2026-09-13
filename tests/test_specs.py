"""The parameterizations: layout, links, and how a vector becomes accumulator quantities.

These are the claims the rest of the codebase makes positionally and would otherwise leave
unchecked — that ``t0`` is last and log-linked, that accumulator 1 is the target, that the
conflict pulse rides the distractor and nothing else.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from eamax.accumulators import SimulatedPulsedWald, Wald
from eamax.design import build_params_fn

from confrdm_jax.specs import (
    CRDM_PARAM_NAMES,
    NUM_CENTERED,
    RDM_PARAM_NAMES,
    crdm_spec,
    make_design,
    rdm_spec,
    spec_for,
)


@pytest.mark.parametrize(
    ("builder", "names"), [(rdm_spec, RDM_PARAM_NAMES), (crdm_spec, CRDM_PARAM_NAMES)]
)
def test_layout_matches_declared_names(builder, names):
    spec = builder()
    assert spec.names == names
    assert spec.num_params == len(names)


@pytest.mark.parametrize("builder", [rdm_spec, crdm_spec])
def test_every_parameter_is_log_linked(builder):
    # The single-subject log-prior adds `sum(x)` as the whole change of variables, and the
    # hierarchical prior is a normal on the log scale. Both assume this.
    assert set(builder().links) == {"log"}


@pytest.mark.parametrize("builder", [rdm_spec, crdm_spec])
def test_t0_is_last_and_centered(builder):
    # T0Support locates t0 by name and raises if it is not log-linked; the hierarchical
    # prior centres the *trailing* num_centered parameters, which must be b and t0.
    spec = builder()
    assert spec.names[-1] == "t0"
    assert spec.names[-2] == "b"
    assert spec.num_centered == NUM_CENTERED


@pytest.mark.parametrize("builder", [rdm_spec, crdm_spec])
def test_responses_are_zero_indexed(builder):
    # The simulators emit choice in {0, 1} and the likelihood reads the same column.
    assert builder().first_response == 0


def test_spec_for_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown model spec"):
        spec_for("wald")


def test_make_design_fills_target_and_distractor():
    data = jnp.array([[0.5, 1.0, 1.0], [0.6, 0.0, 0.0], [0.7, 1.0, 1.0]])
    design = make_design(data)

    # The target accumulator is index 1 on every trial, given as a scalar so eamax treats
    # the quantities built from it as trial-invariant.
    assert design.target.shape == ()
    assert int(design.target) == 1
    # Congruent (condition 1) puts the distracting feature on the target accumulator.
    np.testing.assert_array_equal(design.distractor, data[:, 2])
    np.testing.assert_array_equal(design.response, data[:, 1])
    assert design.first_response == 0


def test_make_design_without_a_condition_column():
    design = make_design(jnp.array([[0.5, 1.0], [0.6, 0.0]]))
    assert design.distractor is None


def test_rdm_quantities():
    theta = jnp.log(jnp.array([1.0, 1.5, 1.2, 2.0, 0.3]))
    spec = rdm_spec()
    params, t0 = build_params_fn(spec, Wald())(theta, make_design(jnp.array([[0.5, 1.0]])))

    # Accumulator 0 is the non-target: drift v_intercept, noise pinned to 1.
    assert params["v"][0, 0] == pytest.approx(1.0)
    assert params["s"][0, 0] == pytest.approx(1.0)
    # Accumulator 1 is the target: drift v_intercept + v_slope, noise s_true.
    assert params["v"][1, 0] == pytest.approx(2.5)
    assert params["s"][1, 0] == pytest.approx(1.2)
    # The boundary is shared, and t0 is a race-level shift rather than a per-accumulator one.
    assert params["b"][0, 0] == pytest.approx(2.0)
    assert params["b"][1, 0] == pytest.approx(2.0)
    assert float(jnp.ravel(t0)[0]) == pytest.approx(0.3)


def test_noise_scale_is_configurable():
    theta = jnp.log(jnp.array([1.0, 1.5, 1.2, 2.0, 0.3]))
    spec = rdm_spec(noise_scale=0.5)
    params, _ = build_params_fn(spec, Wald())(theta, make_design(jnp.array([[0.5, 1.0]])))
    assert params["s"][0, 0] == pytest.approx(0.5)
    assert params["s"][1, 0] == pytest.approx(1.2)


def test_crdm_pulse_rides_only_the_distractor():
    theta = jnp.log(jnp.array([1.0, 4.0, 0.3, 0.1, 1.0, 0.8, 0.3]))
    spec = crdm_spec()
    accumulator = SimulatedPulsedWald(dt=0.01, t_max=1.0)
    # Trial 0 congruent (distractor = target = accumulator 1), trial 1 incongruent.
    data = jnp.array([[0.5, 1.0, 1.0], [0.6, 0.0, 0.0]])
    params, _ = build_params_fn(spec, accumulator)(theta, make_design(data))

    np.testing.assert_allclose(params["amp"][:, 0], [0.0, 0.3], atol=0)
    np.testing.assert_allclose(params["amp"][:, 1], [0.3, 0.0], atol=0)
    # tau is shared: it describes the pulse's shape, not which accumulator carries it.
    np.testing.assert_allclose(params["tau"], 0.1)


def test_crdm_drift_and_noise_match_the_rdm():
    # The conflict model adds a pulse and changes nothing else, so the drift and noise
    # assignments must be identical to the RDM's.
    data = jnp.array([[0.5, 1.0, 1.0]])
    rdm_params, rdm_t0 = build_params_fn(rdm_spec(), Wald())(
        jnp.log(jnp.array([1.0, 1.5, 1.2, 2.0, 0.3])), make_design(data[:, :2])
    )
    crdm_params, crdm_t0 = build_params_fn(
        crdm_spec(), SimulatedPulsedWald(dt=0.01, t_max=1.0)
    )(jnp.log(jnp.array([1.0, 1.5, 0.3, 0.1, 1.2, 2.0, 0.3])), make_design(data))

    for name in ("v", "s", "b"):
        np.testing.assert_allclose(crdm_params[name], rdm_params[name])
    np.testing.assert_allclose(crdm_t0, rdm_t0)


def test_hyperparameters_are_carried_for_the_hierarchical_prior():
    spec = rdm_spec(hyperparameters={"mu_loc": [0.1, 0.3, 0.4, 0.4, -1.2]})
    np.testing.assert_allclose(
        spec.hyperparameter_array("mu_loc"), [0.1, 0.3, 0.4, 0.4, -1.2]
    )
