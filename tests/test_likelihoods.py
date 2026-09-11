"""The race likelihoods: what they score, and what they do with the awkward trials.

The race itself is `eamax`'s and tested there. What is tested here is this repository's
wiring of it — that the analytic RDM likelihood really is the two-accumulator inverse
Gaussian race, that the CRDM's hybrid routes the pulsed density to the accumulator the
design says carries the pulse, and that censored, masked and impossible trials are handled
the way the config comments claim.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from eamax.accumulators import inv_gauss_logpdf, inv_gauss_logsf
from eamax.numerics import MIN_P

from confrdm_jax.likelihoods import (
    create_crdm_hierarchical_likelihood_factory_approx,
    create_crdm_likelihood_factory_approx,
    create_crdm_likelihood_volterra,
    create_rdm_hierarchical_likelihood,
    create_rdm_hierarchical_likelihood_factory_approx,
    create_rdm_likelihood_factory_approx,
    create_rdm_two_accumulators_likelihood,
)
from confrdm_jax.simulators import simulate_crdm, simulate_rdm

LOG_FLOOR = float(jnp.log(MIN_P))


def _reference_rdm_loglik(data, theta):
    """The two-accumulator race, written out by hand as a cross-check.

    Winner's density times loser's survival, with the non-target accumulator at drift
    ``v_intercept`` and noise 1, and the target at ``v_intercept + v_slope`` and noise
    ``s_true``.
    """
    v_i, v_s, s_true, b, t0 = jnp.exp(theta)
    rt, choice = data[:, 0], data[:, 1]
    decision_time = rt - t0

    def parts(v, s):
        mu, lam = b / v, (b / s) ** 2
        return inv_gauss_logpdf(decision_time, mu, lam), inv_gauss_logsf(decision_time, mu, lam)

    pdf_false, sf_false = parts(v_i, 1.0)
    pdf_true, sf_true = parts(v_i + v_s, s_true)

    total = jnp.where(choice == 1, pdf_true + sf_false, pdf_false + sf_true)
    return jnp.maximum(total, LOG_FLOOR)


def test_analytic_rdm_is_the_inverse_gaussian_race(rdm_theta):
    data = simulate_rdm(jax.random.key(0), rdm_theta, 400)
    np.testing.assert_allclose(
        np.asarray(create_rdm_two_accumulators_likelihood(data)(rdm_theta)),
        np.asarray(_reference_rdm_loglik(data, rdm_theta)),
        rtol=1e-12,
    )


def test_analytic_rdm_peaks_near_the_generating_parameters(rdm_theta):
    # Not a recovery test — just enough to catch a likelihood wired to the wrong accumulator,
    # which would still produce finite, plausible-looking numbers.
    data = simulate_rdm(jax.random.key(1), rdm_theta, 4000)
    loglik = create_rdm_two_accumulators_likelihood(data)
    at_truth = float(jnp.sum(loglik(rdm_theta)))
    for index in range(5):
        for factor in (0.8, 1.25):
            perturbed = rdm_theta.at[index].add(jnp.log(factor))
            assert float(jnp.sum(loglik(perturbed))) < at_truth


def test_impossible_trials_land_on_the_flat_floor(rdm_theta):
    # `rt <= t0` used to take a slope-1e3 penalty, so one such trial outweighed a thousand
    # real ones. eamax evaluates it at the clamped decision time and floors the total, which
    # is flat: the likelihood says nothing about which way t0 should move. Keeping out of
    # this region is `T0Support`'s job, at initialisation.
    data = simulate_rdm(jax.random.key(2), rdm_theta, 50)
    theta = rdm_theta.at[4].set(jnp.log(float(jnp.max(data[:, 0])) + 1.0))
    values = create_rdm_two_accumulators_likelihood(data)(theta)
    np.testing.assert_allclose(np.asarray(values), LOG_FLOOR)


def test_neural_rdm_scores_both_accumulators(rdm_theta, wald_conditioner):
    # The Wald flow replaces the closed form on *both* accumulators, so swapping which
    # response won must change the per-trial value of the trials that changed and no others.
    data = simulate_rdm(jax.random.key(3), rdm_theta, 200)
    loglik = create_rdm_likelihood_factory_approx(wald_conditioner)(data)
    values = loglik(rdm_theta)
    assert values.shape == (200,)
    assert jnp.all(jnp.isfinite(values))

    flipped = data.at[:, 1].set(1.0 - data[:, 1])
    flipped_values = create_rdm_likelihood_factory_approx(wald_conditioner)(flipped)(rdm_theta)
    assert not bool(jnp.allclose(values, flipped_values))


def test_crdm_pulse_follows_the_condition_column(crdm_theta, crdm_conditioner):
    # The hybrid gives the pulsed density to whichever accumulator the design's `distractor`
    # column names, which under this design is the congruency indicator. Flipping congruency
    # while holding rt and choice fixed must therefore change the score — if it does not, the
    # routing is not reading the column at all.
    data = simulate_crdm(jax.random.key(4), crdm_theta, 200, dt=0.01, t_max=2.0)
    factory = create_crdm_likelihood_factory_approx(crdm_conditioner, t_max=2.0)

    values = factory(data)(crdm_theta)
    flipped = factory(data.at[:, 2].set(1.0 - data[:, 2]))(crdm_theta)

    assert values.shape == (200,)
    assert jnp.all(jnp.isfinite(values))
    assert not bool(jnp.allclose(values, flipped))


def test_crdm_reduces_to_the_rdm_as_the_pulse_vanishes(rdm_theta):
    # The Volterra solver is the reference the flow approximates, and it reduces to the
    # inverse Gaussian when there is no pulse. With amp -> 0 the whole hybrid — the routing,
    # the gather/overlay round trip, the race assembly — must reproduce the analytic RDM on
    # the same data. That is the end-to-end check of the hybrid that an untrained flow
    # cannot give.
    data = simulate_rdm(jax.random.key(5), rdm_theta, 40)
    with_condition = jnp.concatenate([data, jnp.ones((data.shape[0], 1))], axis=-1)

    v_i, v_s, s_true, b, t0 = jnp.exp(rdm_theta)
    crdm_theta = jnp.log(jnp.array([v_i, v_s, 1e-8, 0.1, s_true, b, t0]))

    volterra = create_crdm_likelihood_volterra(with_condition, dt=0.002, t_max=3.0)
    np.testing.assert_allclose(
        np.asarray(volterra(crdm_theta)),
        np.asarray(create_rdm_two_accumulators_likelihood(data)(rdm_theta)),
        atol=2e-3,
    )


def test_censored_trials_are_scored_by_survival(crdm_theta, crdm_conditioner):
    # A trial marked rt = -1.0 is the observation `T > t_max`, not missing data: its
    # likelihood is the probability that *every* accumulator was still running. Without
    # `t_max` the sentinel is just an infeasible trial and lands on the floor.
    data = simulate_crdm(jax.random.key(6), crdm_theta, 20, dt=0.01, t_max=2.0)
    censored = data.at[0, 0].set(-1.0).at[0, 1].set(-1.0)

    scored = create_crdm_likelihood_factory_approx(crdm_conditioner, t_max=2.0)
    unscored = create_crdm_likelihood_factory_approx(crdm_conditioner, t_max=None)

    censored_value = float(scored(censored)(crdm_theta)[0])
    assert LOG_FLOOR < censored_value < 0.0

    # Without `t_max` the same trial is evaluated at the clamped decision time instead, where
    # every accumulator has survived with probability ~1 — so it contributes ~0 and is
    # silently *ignored* rather than scored. That is the failure mode the config's
    # `t_max: ${model.test_sampler.t_max}` interpolation exists to rule out, and it is
    # invisible in a log rather than obviously wrong.
    assert float(unscored(censored)(crdm_theta)[0]) == pytest.approx(0.0, abs=1e-12)

    # Every other trial is untouched by the substitution.
    np.testing.assert_allclose(
        np.asarray(scored(censored)(crdm_theta)[1:]),
        np.asarray(scored(data)(crdm_theta)[1:]),
    )


def test_hierarchical_rdm_sums_over_subjects(rdm_theta):
    thetas = jnp.stack([rdm_theta, rdm_theta + 0.1, rdm_theta - 0.1])
    data = jnp.stack([
        simulate_rdm(jax.random.key(seed), theta, 100)
        for seed, theta in enumerate(thetas)
    ])
    mask = jnp.ones((3, 100), dtype=bool)

    total = create_rdm_hierarchical_likelihood(data, mask)(thetas)
    per_subject = sum(
        float(jnp.sum(create_rdm_two_accumulators_likelihood(data[i])(thetas[i])))
        for i in range(3)
    )
    assert float(total) == pytest.approx(per_subject, rel=1e-12)


def test_masked_trials_contribute_exactly_zero(rdm_theta):
    # The mask exists so subjects with differing trial counts can be padded to a rectangle.
    # Padding must contribute 0 whatever it holds — including values that would otherwise be
    # NaN, which is the case the test uses.
    thetas = jnp.stack([rdm_theta, rdm_theta])
    data = jnp.stack([simulate_rdm(jax.random.key(i), rdm_theta, 60) for i in range(2)])
    padded = data.at[1, 30:, :].set(jnp.nan)

    mask = jnp.ones((2, 60), dtype=bool).at[1, 30:].set(False)
    masked_total = create_rdm_hierarchical_likelihood(padded, mask)(thetas)

    expected = float(jnp.sum(create_rdm_two_accumulators_likelihood(data[0])(rdm_theta))) + float(
        jnp.sum(create_rdm_two_accumulators_likelihood(data[1][:30])(rdm_theta))
    )
    assert float(masked_total) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("remat", [False, True])
def test_hierarchical_neural_factories_run(remat, rdm_theta, crdm_theta,
                                           wald_conditioner, crdm_conditioner):
    # `remat` trades recomputation for activation memory and must not change the value.
    thetas = jnp.stack([rdm_theta, rdm_theta + 0.05])
    data = jnp.stack([simulate_rdm(jax.random.key(i), rdm_theta, 50) for i in range(2)])
    mask = jnp.ones((2, 50), dtype=bool)
    value = create_rdm_hierarchical_likelihood_factory_approx(wald_conditioner, remat=remat)(
        data, mask
    )(thetas)
    assert jnp.isfinite(value)

    crdm_thetas = jnp.stack([crdm_theta, crdm_theta + 0.05])
    crdm_data = jnp.stack([
        simulate_crdm(jax.random.key(i), crdm_theta, 50, dt=0.01, t_max=2.0) for i in range(2)
    ])
    crdm_value = create_crdm_hierarchical_likelihood_factory_approx(
        crdm_conditioner, t_max=2.0, remat=remat
    )(crdm_data, mask)(crdm_thetas)
    assert jnp.isfinite(crdm_value)


def test_gradients_are_finite(rdm_theta, wald_conditioner):
    # NUTS differentiates through this, and a NaN anywhere in the guard logic poisons the
    # whole gradient rather than one term.
    data = simulate_rdm(jax.random.key(7), rdm_theta, 200)
    loglik = create_rdm_likelihood_factory_approx(wald_conditioner)(data)
    grad = jax.grad(lambda x: jnp.sum(loglik(x)))(rdm_theta)
    assert jnp.all(jnp.isfinite(grad))
