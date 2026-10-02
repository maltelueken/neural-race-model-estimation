"""Priors — the part of the model that stays in this repository.

Two things are worth pinning: that the distributions are what the config claims (a truncated
normal really is truncated, a gamma's scale means what the docstring says), and that the
change of variables between the sampler's log space and the prior's natural space is right.
A wrong Jacobian does not raise; it tilts the posterior.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from eamax.hierarchical import HierarchicalFlatSpace, HierarchicalLKJMVNPrior
from scipy import stats as sps

from confrdm_jax.priors import (
    TruncatedNormal,
    create_crdm_prior_informed,
    create_hierarchical_crdm_prior_lkj_mvn,
    create_hierarchical_rdm_prior_lkj_mvn,
    create_rdm_prior_informed,
    log_prior_fn,
)
from confrdm_jax.specs import CRDM_PARAM_NAMES, RDM_PARAM_NAMES


def test_truncated_normal_log_prob_matches_scipy():
    dist = TruncatedNormal(0.3, 0.2, 0.0, jnp.inf)
    x = jnp.linspace(0.01, 2.0, 50)
    expected = sps.truncnorm.logpdf(np.asarray(x), a=(0.0 - 0.3) / 0.2, b=np.inf, loc=0.3, scale=0.2)
    np.testing.assert_allclose(np.asarray(dist.log_prob(x)), expected, rtol=1e-10)


def test_truncated_normal_samples_stay_inside_the_bounds():
    # Inverse-CDF sampling, so this holds exactly rather than with high probability — which
    # matters because a draw below 0 would become NaN the moment the recovery script logs it.
    dist = TruncatedNormal(0.1, 1.0, 0.0, jnp.inf)
    draws = dist.sample(seed=jax.random.key(0), sample_shape=(20000,))
    assert float(jnp.min(draws)) > 0.0


@pytest.mark.parametrize(
    ("builder", "names"),
    [(create_rdm_prior_informed, RDM_PARAM_NAMES), (create_crdm_prior_informed, CRDM_PARAM_NAMES)],
)
def test_recovery_priors_are_positive_and_the_right_width(builder, names):
    # Every parameter is log-linked, so a non-positive draw would be NaN in log space.
    draws = jnp.stack(builder().sample(seed=jax.random.key(1), sample_shape=(5000,)), axis=-1)
    assert draws.shape == (5000, len(names))
    assert float(jnp.min(draws)) > 0.0


def test_gamma_scale_is_a_scale_not_a_rate():
    # `s_true_shape=12, s_true_scale=0.1` is documented to mean a mean of 1.2, not 12. The
    # factory passes `rate = 1 / scale`, and getting that backwards would move the prior by
    # two orders of magnitude without raising.
    prior = create_rdm_prior_informed(s_true_shape=12.0, s_true_scale=0.1)
    s_true = prior.sample(seed=jax.random.key(2), sample_shape=(50000,))[2]
    assert float(jnp.mean(s_true)) == pytest.approx(1.2, rel=0.02)


def test_log_prior_includes_the_log_transform_jacobian():
    # The sampler works in log space while the prior is defined on the natural scale, so the
    # density must carry d(exp(x))/dx = exp(x), i.e. + sum(x).
    prior = create_rdm_prior_informed()
    log_prior = log_prior_fn(prior)
    x = jnp.log(jnp.array([1.0, 1.5, 1.2, 1.2, 0.3]))

    natural = jnp.exp(x)
    expected = prior.log_prob([natural[i] for i in range(5)]) + jnp.sum(x)
    assert float(log_prior(x)) == pytest.approx(float(expected), rel=1e-12)


def test_log_prior_integrates_to_one_in_log_space():
    # A one-dimensional slice is enough to catch a missing, doubled or wrong-signed Jacobian:
    # with the change of variables right, the log-space density integrates to 1 over the
    # real line. Everything but `t0` is held fixed and contributes a constant, recovered from
    # a reference point so it can be divided back out.
    prior = create_rdm_prior_informed(t0_loc=0.3, t0_scale=0.2)
    log_prior = log_prior_fn(prior)
    others = jnp.log(jnp.array([1.0, 1.5, 1.2, 1.2]))

    grid = jnp.linspace(-20.0, 6.0, 120001)
    slice_density = jax.vmap(lambda t: log_prior(jnp.append(others, t)))(grid)

    # exp(constant) = exp(log_prior at the reference) / (p_t0(0.3) * 0.3), the second factor
    # being the t0 component's density times its own Jacobian.
    reference = log_prior(jnp.append(others, jnp.log(0.3)))
    t0_factor = prior.distributions[-1].prob(0.3) * 0.3
    constant = jnp.exp(reference) / t0_factor

    total = jnp.trapezoid(jnp.exp(slice_density), grid) / constant
    assert float(total) == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize(
    ("builder", "names"),
    [
        (create_hierarchical_rdm_prior_lkj_mvn, RDM_PARAM_NAMES),
        (create_hierarchical_crdm_prior_lkj_mvn, CRDM_PARAM_NAMES),
    ],
)
def test_hierarchical_prior_matches_its_spec(builder, names):
    num_params = len(names)
    prior = builder(
        6,
        inverse_gamma_scale=jnp.ones(num_params),
        mu_loc=jnp.zeros(num_params),
        mu_scale=jnp.ones(num_params),
    )
    assert isinstance(prior, HierarchicalLKJMVNPrior)
    assert prior.num_params == num_params
    # b and t0 are the centered block; everything before them is non-centered.
    assert prior.num_centered == 2
    assert prior.num_params_ncp == num_params - 2

    draw = prior.sample(seed=jax.random.key(3))
    assert draw["mu"].shape == (num_params,)
    assert draw["psi_raw"].shape == (num_params, num_params)
    assert draw["z"].shape == (6, num_params - 2)
    assert draw["theta_bt"].shape == (6, 2)


def test_hierarchical_prior_rejects_the_wrong_models_hyperparameters():
    # The RDM's five-entry arrays in the CRDM's seven-parameter prior is exactly the mistake
    # a copied config block makes, and it has to fail rather than broadcast.
    with pytest.raises(ValueError, match="shape"):
        create_hierarchical_crdm_prior_lkj_mvn(
            4, inverse_gamma_scale=jnp.ones(5), mu_loc=jnp.zeros(5), mu_scale=jnp.ones(5)
        )


def test_flat_space_round_trips_and_carries_the_jacobian():
    prior = create_hierarchical_rdm_prior_lkj_mvn(
        3,
        inverse_gamma_scale=jnp.full(5, 1.6),
        mu_loc=jnp.zeros(5),
        mu_scale=jnp.full(5, 0.15),
    )
    flat_space = prior.flat_space()
    assert isinstance(flat_space, HierarchicalFlatSpace)

    flat = flat_space.sample(jax.random.key(4))
    assert flat.shape == (flat_space.num_flat_params,)

    # The reconstruction is the formula every consumer used to write out for itself.
    subject_params = flat_space.subject_params(flat)
    assert subject_params.shape == (3, 5)
    # The centered block passes through unchanged, which is what lets T0Support test `t0`
    # without running the whole reconstruction.
    np.testing.assert_allclose(flat_space.centered_block(flat), subject_params[:, -2:])

    # log_prob is the constrained density plus the change of variables; the split is what
    # `eamax.hierarchical.joint_log_det_jacobian` computes component by component.
    constrained = flat_space.forward(flat)
    expected = prior.log_prob(constrained) + flat_space.log_det_jacobian(flat)
    assert float(flat_space.log_prob(flat)) == pytest.approx(float(expected), rel=1e-12)
