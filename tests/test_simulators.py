"""Simulators: shapes, the sentinel, and agreement with the distribution being scored.

The generative side has to agree with the likelihood or every recovery result is measuring
the wrong thing. The strongest check here is the score test — the gradient of the
log-likelihood at the generating parameters averages to zero over simulated datasets — which
ties each simulator to its own likelihood rather than to a hand-written formula.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from eamax.hierarchical import reconstruct_from_dict

from confrdm_jax.likelihoods import create_rdm_two_accumulators_likelihood
from confrdm_jax.simulators import (
    crdm_condition_design,
    sample_conditional_crdm_condition,
    sample_conditional_crdm_hierarchical_lkj_mvn,
    sample_conditional_crdm_single,
    sample_conditional_rdm,
    sample_conditional_rdm_hierarchical_lkj_mvn,
    sample_conditional_wald,
    simulate_crdm,
    simulate_rdm,
)
from confrdm_jax.specs import CRDM_PARAM_NAMES, RDM_PARAM_NAMES

HIERARCHICAL_PRIOR_CFG = {
    "inverse_gamma_concentration": 15.0,
    "lkj_concentration": 2.0,
}


def _hierarchical_kwargs(num_params):
    return {
        **HIERARCHICAL_PRIOR_CFG,
        "inverse_gamma_scale": [1.6] * num_params,
        "mu_loc": [0.1] * (num_params - 1) + [-1.2],
        "mu_scale": [0.15] * num_params,
    }


# --------------------------------------------------------------------------------------- #
# Training samplers
# --------------------------------------------------------------------------------------- #


def test_wald_training_sampler_shapes(wald_training_prior):
    data, context = sample_conditional_wald(jax.random.key(0), (8, 50), wald_training_prior)
    # (B, T, 1) decision times and (B, 1, num_in) context: the middle context axis is a
    # broadcasting axis, one set of spline knots per parameter draw.
    assert data.shape == (8, 50, 1)
    assert context.shape == (8, 1, 3)
    # Exact inverse Gaussian draws, so this sampler can never censor.
    assert bool(jnp.all(jnp.isfinite(data)))
    assert float(jnp.min(data)) > 0.0


def test_wald_training_sampler_draws_the_inverse_gaussian(wald_training_prior):
    # The flow is fitted to these draws, so they have to be the distribution the analytic
    # likelihood scores. mean = b/v and var = b*s^2/v^3 for the first-passage time.
    from eamax.accumulators import Wald

    params = {"v": jnp.full(200000, 2.0), "s": jnp.ones(200000), "b": jnp.full(200000, 1.5)}
    draws = Wald().sample(jax.random.key(1), params)
    assert float(jnp.mean(draws)) == pytest.approx(1.5 / 2.0, rel=0.01)
    assert float(jnp.var(draws)) == pytest.approx(1.5 * 1.0 / 2.0**3, rel=0.05)


def test_crdm_training_sampler_shapes_and_censoring(crdm_training_prior):
    data, context = sample_conditional_crdm_single(
        jax.random.key(2), (8, 40), crdm_training_prior, dt=0.01, t_max=1.0
    )
    assert data.shape == (8, 40, 1)
    assert context.shape == (8, 1, 5)
    # A non-crossing accumulator returns inf rather than a negative sentinel; there is no
    # race here to turn it into one, and `eamax.flows.loss_fn` reads a non-finite decision
    # time as censored and scores it by log S(t_max).
    finite = data[jnp.isfinite(data)]
    assert float(jnp.min(finite)) > 0.0
    assert bool(jnp.all(finite <= 1.0))


def test_crdm_training_sampler_censors_when_the_horizon_is_short():
    # Slow drift and a high boundary is the corner where censoring concentrates; the
    # censored fraction has to reach the loss function rather than being dropped.
    import distrax

    prior = distrax.Joint([
        distrax.Uniform(0.05, 0.1),   # v_c
        distrax.Uniform(0.0, 0.01),   # amp
        distrax.Uniform(0.05, 0.1),   # tau
        distrax.Uniform(0.1, 0.2),    # s
        distrax.Uniform(2.0, 3.0),    # b
    ])
    data, _ = sample_conditional_crdm_single(
        jax.random.key(3), (4, 50), prior, dt=0.01, t_max=0.5
    )
    assert bool(jnp.any(~jnp.isfinite(data)))


# --------------------------------------------------------------------------------------- #
# Test samplers
# --------------------------------------------------------------------------------------- #


def test_rdm_test_sampler_shapes(rdm_prior):
    data, context = sample_conditional_rdm(jax.random.key(4), (5, 60), rdm_prior)
    assert data.shape == (5, 60, 2)
    assert context.shape == (5, len(RDM_PARAM_NAMES))
    assert set(np.unique(np.asarray(data[:, :, 1]))) <= {0.0, 1.0}
    # No censoring is possible, so no sentinel.
    assert float(jnp.min(data[:, :, 0])) > 0.0


def test_crdm_test_sampler_shapes_and_condition_column(crdm_prior):
    data, context = sample_conditional_crdm_condition(
        jax.random.key(5), (3, 40), crdm_prior, dt=0.005, t_max=2.0
    )
    assert data.shape == (3, 40, 3)
    assert context.shape == (3, len(CRDM_PARAM_NAMES))
    # Congruent first half, incongruent second: the likelihood reads congruency from this
    # column and routes the conflict pulse by it.
    np.testing.assert_array_equal(np.asarray(data[0, :20, 2]), np.ones(20))
    np.testing.assert_array_equal(np.asarray(data[0, 20:, 2]), np.zeros(20))


def test_crdm_condition_design_splits_congruency():
    design = crdm_condition_design(7)
    # An odd count gives the extra trial to the congruent half.
    np.testing.assert_array_equal(np.asarray(design.condition), [1, 1, 1, 1, 0, 0, 0])
    # `distractor` names the accumulator carrying the pulse, and for this design that is the
    # congruency indicator itself.
    np.testing.assert_array_equal(np.asarray(design.distractor), np.asarray(design.condition))


def test_conflict_slows_and_impairs_responses(crdm_theta):
    # The substantive claim the model exists to express. Incongruent trials put the pulse on
    # the non-target accumulator, so they should be slower and less accurate.
    data = simulate_crdm(jax.random.key(6), crdm_theta, 6000, dt=0.002, t_max=3.0)
    congruent, incongruent = data[:3000], data[3000:]

    assert float(jnp.mean(congruent[:, 0])) < float(jnp.mean(incongruent[:, 0]))
    assert float(jnp.mean(congruent[:, 1] == 1)) > float(jnp.mean(incongruent[:, 1] == 1))


def test_rdm_score_test(rdm_theta):
    # The gradient of the log-likelihood at the generating parameters has expectation zero
    # when the simulator and the likelihood describe the same distribution. This is what
    # catches a simulator and a likelihood that disagree about, say, which accumulator
    # carries `s_true` — a mistake that leaves both of them individually plausible.
    def score(key):
        data = simulate_rdm(key, rdm_theta, 400)
        return jax.grad(
            lambda x: jnp.sum(create_rdm_two_accumulators_likelihood(data)(x))
        )(rdm_theta)

    scores = jax.vmap(score)(jax.random.split(jax.random.key(7), 200))
    mean = np.asarray(jnp.mean(scores, axis=0))
    standard_error = np.asarray(jnp.std(scores, axis=0)) / np.sqrt(scores.shape[0])
    np.testing.assert_array_less(np.abs(mean), 4.0 * standard_error)


# --------------------------------------------------------------------------------------- #
# Hierarchical samplers
# --------------------------------------------------------------------------------------- #


def test_hierarchical_rdm_sampler():
    data, context = sample_conditional_rdm_hierarchical_lkj_mvn(
        jax.random.key(8), 30, 6, **_hierarchical_kwargs(5)
    )
    assert data.shape == (6, 30, 2)
    assert set(context) == {"s", "mu", "psi_raw", "z", "theta_bt"}

    # The reconstruction the sampler used is the one `eamax.hierarchical` defines, and the
    # same one the likelihood wrapper and the post-processing reach through — which is the
    # point of it living in one place.
    log_theta = reconstruct_from_dict(context)
    assert log_theta.shape == (6, 5)
    # The centered block passes through untouched.
    np.testing.assert_allclose(log_theta[:, -2:], context["theta_bt"])


def test_hierarchical_crdm_sampler():
    data, context = sample_conditional_crdm_hierarchical_lkj_mvn(
        jax.random.key(9), 20, 4, dt=0.01, t_max=2.0, **_hierarchical_kwargs(7)
    )
    assert data.shape == (4, 20, 3)
    assert reconstruct_from_dict(context).shape == (4, 7)
    np.testing.assert_array_equal(np.asarray(data[0, :10, 2]), np.ones(10))
    np.testing.assert_array_equal(np.asarray(data[0, 10:, 2]), np.zeros(10))


def test_hierarchical_subjects_differ():
    # Shrinkage is part of what recovery measures, so the subjects have to be genuinely
    # different draws rather than one draw repeated.
    _, context = sample_conditional_rdm_hierarchical_lkj_mvn(
        jax.random.key(10), 20, 8, **_hierarchical_kwargs(5)
    )
    log_theta = reconstruct_from_dict(context)
    assert float(jnp.min(jnp.std(log_theta, axis=0))) > 0.0
