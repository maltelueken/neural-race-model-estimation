"""Wiring of `eamax.inference` into this repository's models.

The samplers are `eamax`'s and tested there. What is tested here is what this repository
hands them: that the ``t0`` support constraint finds the right parameter in these specs, that
starting positions land inside the support and stay dispersed, and that a hierarchical
particle is understood the same way by the constraint, the log-density and the
post-processing.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from eamax.inference import (
    T0Support,
    init_particles_from_prior,
    init_positions_from_prior,
    min_valid_rt,
)

from confrdm_jax.priors import create_hierarchical_rdm_prior_lkj_mvn, log_prior_fn
from confrdm_jax.simulators import simulate_rdm
from confrdm_jax.specs import crdm_spec, rdm_spec


@pytest.fixture
def hierarchical_prior():
    return create_hierarchical_rdm_prior_lkj_mvn(
        5,
        inverse_gamma_scale=jnp.full(5, 1.6),
        mu_loc=jnp.array([0.1, 0.3, 0.4, 0.4, -1.2]),
        mu_scale=jnp.array([0.15, 0.15, 0.15, 0.15, 0.1]),
    )


@pytest.mark.parametrize("builder", [rdm_spec, crdm_spec])
def test_t0_support_locates_t0_by_name(builder):
    spec = builder()
    support = T0Support.from_spec(spec, 0.5, max_fraction=0.97)
    assert support.index == spec.index("t0")
    assert float(support.log_t0_max) == pytest.approx(float(jnp.log(0.97 * 0.5)))


def test_t0_support_rejects_a_spec_without_t0():
    # Both specs put t0 last on the log link and several call sites depend on it. The point
    # of locating it by name is that a layout change fails here instead of silently clipping
    # whichever parameter happens to be last.
    from eamax.design import Parameterization

    spec = Parameterization.of_names(["v", "b"], ["log", "log"])
    with pytest.raises(ValueError, match="t0"):
        T0Support.from_spec(spec, 0.5)


def test_min_valid_rt_excludes_the_censoring_sentinel():
    # Taking the raw minimum would give -1.0, whose log is NaN, and the constraint would then
    # be satisfied by nothing — silently disabling itself for that subject.
    rt = jnp.array([-1.0, 0.42, 0.55, -1.0])
    assert float(min_valid_rt(rt)) == pytest.approx(0.42)


def test_min_valid_rt_raises_when_a_subject_has_no_valid_trial():
    with pytest.raises(ValueError, match="No valid response time"):
        min_valid_rt(jnp.array([[-1.0, -1.0], [0.4, 0.5]]))


def test_starting_positions_are_in_support_and_dispersed(rdm_prior, rdm_theta):
    data = simulate_rdm(jax.random.key(0), rdm_theta, 300)
    spec = rdm_spec()
    support = T0Support.from_spec(spec, min_valid_rt(data[:, 0]), max_fraction=0.97)

    positions, num_exhausted = init_positions_from_prior(
        lambda key: jnp.log(jnp.stack(rdm_prior.sample(seed=key))),
        64, jax.random.key(1), support=support, max_attempts=2000,
    )

    assert positions.shape == (64, 5)
    assert int(num_exhausted) == 0
    # Every start clears the cap: a chain that begins above `min(rt)` starts on the
    # likelihood's flat floor, where the gradient carries no information and window
    # adaptation cannot recover.
    assert bool(jnp.all(positions[:, spec.index("t0")] <= support.log_t0_max))
    # Rejection, not clipping — a clip is a point mass in the one coordinate the dispersion
    # exists to spread, so the accepted t0 values must still take many distinct values.
    t0_values = positions[:, spec.index("t0")]
    assert int(jnp.unique(t0_values).size) > 32
    # Every parameter keeps the prior's dispersion.
    assert float(jnp.min(jnp.std(positions, axis=0))) > 0.0


def test_hierarchical_particles_respect_every_subjects_own_bound(hierarchical_prior):
    # `t0` is a subject-level parameter, so the cap is per subject: pooling would let a fast
    # subject's floor license a start above a slow subject's fastest trial.
    flat_space = hierarchical_prior.flat_space()
    min_rt = jnp.array([0.30, 0.45, 0.60, 0.35, 0.50])
    support = T0Support.from_spec(rdm_spec(), min_rt, max_fraction=0.97)

    particles, num_exhausted = init_particles_from_prior(
        flat_space, 40, jax.random.key(2), support=support, max_attempts=2000,
    )

    assert particles.shape == (40, flat_space.num_flat_params)
    assert int(num_exhausted) == 0

    subject_params = jax.vmap(flat_space.subject_params)(particles)
    assert subject_params.shape == (40, 5, 5)
    assert bool(jnp.all(subject_params[:, :, -1] <= support.log_t0_max[None, :]))


def test_unconstrained_logdensity_is_finite_and_differentiable(hierarchical_prior,
                                                               wald_conditioner):
    # The flat space is what the sampler explores: the prior's change of variables, the
    # semi-centered reconstruction and the likelihood all have to compose there.
    from confrdm_jax.likelihoods import create_rdm_hierarchical_likelihood

    flat_space = hierarchical_prior.flat_space()
    theta = jnp.log(jnp.array([1.0, 1.5, 1.2, 1.2, 0.3]))
    data = jnp.stack([simulate_rdm(jax.random.key(i), theta, 80) for i in range(5)])
    mask = jnp.ones((5, 80), dtype=bool)
    likelihood = create_rdm_hierarchical_likelihood(data, mask)

    def logdensity(flat):
        return flat_space.log_prob(flat) + likelihood(flat_space.subject_params(flat))

    flat = flat_space.sample(jax.random.key(3))
    assert jnp.isfinite(logdensity(flat))
    assert bool(jnp.all(jnp.isfinite(jax.grad(logdensity)(flat))))


def test_single_subject_logdensity_is_differentiable(rdm_prior, rdm_theta):
    from confrdm_jax.likelihoods import create_rdm_two_accumulators_likelihood

    data = simulate_rdm(jax.random.key(4), rdm_theta, 200)
    likelihood = create_rdm_two_accumulators_likelihood(data)
    log_prior = log_prior_fn(rdm_prior)

    grad = jax.grad(lambda x: log_prior(x) + jnp.sum(likelihood(x)))(rdm_theta)
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_degenerate_chains_are_dropped_not_repaired():
    """The migration's one deliberate behaviour change, as a rule on step sizes.

    Repairing a collapsed step size with the healthy chains' median was tried and measured:
    the repaired chains came out under-dispersed at 0.16-0.83x their siblings' spread and
    still broke R-hat (1.94 / 2.60 / 2.76), falling to ~1.01 only once they were dropped.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "recovery_hierarchical", "scripts/parameter_recovery_hierarchical.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    kept, dropped = module.healthy_chains(np.array([0.12, 3e-5, 0.09, 0.11]))
    np.testing.assert_array_equal(kept, [0, 2, 3])
    assert dropped == 1

    kept, dropped = module.healthy_chains(np.array([0.12, 0.09, 0.11, 0.10]))
    np.testing.assert_array_equal(kept, [0, 1, 2, 3])
    assert dropped == 0

    # Every chain collapsed: there is nothing to compare against and nothing to keep, so all
    # are kept and the caller reports the run as unusable rather than returning nothing.
    kept, dropped = module.healthy_chains(np.array([1e-6, 3e-5]))
    np.testing.assert_array_equal(kept, [0, 1])
    assert dropped == 0
