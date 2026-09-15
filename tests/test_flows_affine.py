"""The local affine patch over `eamax.flows`.

What must hold for it to be a safe experiment: a plain conditioner behaves exactly as
`eamax`'s did, the affine stage is exactly ``log T = loc + scale * Y``, the survival
function is still exact given the flow, and a checkpoint cannot be loaded with the wrong
layout.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from eamax.flows import evaluate_pdf_sf as eamax_evaluate_pdf_sf
from eamax.flows import load_conditioner as eamax_load_conditioner
from flax import nnx

from confrdm_jax import flows_affine
from confrdm_jax.flows_affine import (
    MIN_SCALE,
    FlowAccumulator,
    conditioner_layout,
    evaluate_pdf_sf,
    load_conditioner,
    loss_fn,
    make_mlp_conditioner,
    save_conditioner,
    spline_flow,
)
from confrdm_jax.likelihoods import create_crdm_hierarchical_likelihood_factory_approx
from confrdm_jax.simulators import simulate_crdm

NUM_BINS = 4
CONTEXT = jnp.array([[[2.0, 1.0, 1.0]], [[1.0, 1.5, 0.8]]])  # (N, 1, num_in)
TIMES = jnp.tile(jnp.linspace(0.05, 3.0, 40), (2, 1))  # (N, T)


def _pair():
    """A plain conditioner and an affine one that shares its spline weights, affine rows zeroed."""
    plain = make_mlp_conditioner(nnx.Rngs(0), num_in=3, num_mid=16, num_bins=NUM_BINS)
    affine = make_mlp_conditioner(nnx.Rngs(1), num_in=3, num_mid=16, num_bins=NUM_BINS, affine=True)
    width = 3 * NUM_BINS + 1
    affine.linear1.kernel.value = plain.linear1.kernel.value
    affine.linear1.bias.value = plain.linear1.bias.value
    affine.linear2.kernel.value = (
        jnp.zeros_like(affine.linear2.kernel.value).at[:, :width].set(plain.linear2.kernel.value)
    )
    affine.linear2.bias.value = jnp.zeros(width + 2).at[:width].set(plain.linear2.bias.value)
    return plain, affine


def _set_affine_outputs(conditioner, loc, raw_scale):
    conditioner.linear2.bias.value = conditioner.linear2.bias.value.at[-2].set(loc).at[-1].set(raw_scale)


def test_layout_is_read_off_the_output_width():
    plain, affine = _pair()
    assert conditioner_layout(plain) == (NUM_BINS, False)
    assert conditioner_layout(affine) == (NUM_BINS, True)
    with pytest.raises(ValueError, match="neither 3K\\+1"):
        flows_affine._layout_from_width(3 * NUM_BINS + 2)


def test_a_plain_conditioner_is_exactly_eamax():
    plain, _ = _pair()
    ours = evaluate_pdf_sf(plain, TIMES, CONTEXT)
    theirs = eamax_evaluate_pdf_sf(plain, TIMES, CONTEXT)
    for a, b in zip(ours, theirs, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_zeroed_affine_outputs_reproduce_the_plain_flow():
    # loc = 0 and scale = 1 must be the identity, bit for bit, so an affine conditioner
    # starts training from the same family the plain one does.
    plain, affine = _pair()
    for a, b in zip(
        evaluate_pdf_sf(plain, TIMES, CONTEXT), evaluate_pdf_sf(affine, TIMES, CONTEXT), strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_the_affine_stage_is_a_location_and_scale_on_log_time():
    plain, affine = _pair()
    _set_affine_outputs(affine, loc=-1.3, raw_scale=-0.7)
    scale = float(jax.nn.softplus(-0.7 + flows_affine._SCALE_OFFSET) + MIN_SCALE)

    z = jnp.tile(jnp.linspace(-6.0, 6.0, 9), (2, 1))  # beyond the spline range on both sides
    _, plain_flow = spline_flow(TIMES, CONTEXT, plain)
    _, affine_flow = spline_flow(TIMES, CONTEXT, affine)
    np.testing.assert_allclose(
        np.log(np.asarray(affine_flow.bijector.forward(z))),
        -1.3 + scale * np.log(np.asarray(plain_flow.bijector.forward(z))),
        atol=1e-12,
    )


def test_survival_is_exact_given_the_affine_flow():
    # The race multiplies flow densities by flow survivals; that is only sound while every
    # stage is strictly increasing. Check S(t) against the flow's own integrated density.
    _, affine = _pair()
    _set_affine_outputs(affine, loc=-1.0, raw_scale=-0.5)
    context = jnp.array([2.0, 1.0, 1.0])

    grid = jnp.linspace(1e-5, 20.0, 400_000)
    log_pdf, _ = spline_flow(grid, context, affine)
    cumulative = jnp.cumsum(jnp.exp(log_pdf)) * (grid[1] - grid[0])
    assert float(cumulative[-1]) == pytest.approx(1.0, abs=2e-3)

    for target in (0.1, 0.4, 1.0):
        index = int(jnp.searchsorted(grid, target))
        reported = float(jnp.exp(evaluate_pdf_sf(affine, grid[index : index + 1], context)[1][0]))
        assert reported == pytest.approx(1.0 - float(cumulative[index]), abs=2e-3)


def test_loss_gradient_reaches_the_affine_outputs():
    _, affine = _pair()
    context = jnp.tile(jnp.array([2.0, 1.0, 1.0]), (32, 1))
    data = jnp.linspace(0.2, 2.0, 32).at[-1].set(-1.0)  # one censored trial

    grads = jax.grad(lambda c: loss_fn(c, data, context, 4.0))(affine)
    kernel = np.asarray(grads.linear2.kernel.value)
    assert np.all(np.isfinite(kernel))
    assert np.any(kernel[:, -2:] != 0.0)


def test_the_affine_accumulator_runs_in_the_hybrid_race(crdm_theta):
    affine = make_mlp_conditioner(nnx.Rngs(2), num_in=5, num_mid=16, num_bins=NUM_BINS, affine=True)
    assert isinstance(
        create_crdm_hierarchical_likelihood_factory_approx.__globals__["_flow_accumulator"](
            affine, ("v", "amp", "tau", "s", "b")
        ),
        FlowAccumulator,
    )
    thetas = jnp.stack([crdm_theta, crdm_theta + 0.05])
    data = jnp.stack([
        simulate_crdm(jax.random.key(i), crdm_theta, 50, dt=0.01, t_max=2.0) for i in range(2)
    ])
    mask = jnp.ones((2, 50), dtype=bool)
    loglik = create_crdm_hierarchical_likelihood_factory_approx(affine, t_max=2.0)(data, mask)
    value, grad = jax.value_and_grad(loglik)(thetas)
    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(grad)))


def test_an_affine_checkpoint_round_trips_and_records_its_layout(tmp_path):
    _, affine = _pair()
    _set_affine_outputs(affine, loc=-0.8, raw_scale=0.3)
    path = tmp_path / "conditioner"
    save_conditioner(affine, str(path), context_names=("v", "s", "b"), num_mid=16, num_bins=NUM_BINS)

    from eamax.flows.checkpoint import read_metadata

    assert read_metadata(str(path))["affine"] is True

    template = make_mlp_conditioner(nnx.Rngs(9), num_in=3, num_mid=16, num_bins=NUM_BINS, affine=True)
    restored = load_conditioner(template, str(path), context_names=("v", "s", "b"))
    # Parameters restore at the template's float32, hence a tolerance rather than equality.
    np.testing.assert_allclose(
        np.asarray(evaluate_pdf_sf(affine, TIMES, CONTEXT)[0]),
        np.asarray(evaluate_pdf_sf(restored, TIMES, CONTEXT)[0]),
        rtol=1e-6,
    )


def test_loading_with_the_wrong_layout_is_refused(tmp_path):
    plain, affine = _pair()
    for saved, wrong in ((affine, False), (plain, True)):
        path = tmp_path / f"conditioner_{wrong}"
        save_conditioner(saved, str(path), context_names=("v", "s", "b"))
        template = make_mlp_conditioner(
            nnx.Rngs(0), num_in=3, num_mid=16, num_bins=NUM_BINS, affine=wrong
        )
        with pytest.raises(ValueError, match="flow_affine"):
            load_conditioner(template, str(path), context_names=("v", "s", "b"))


def test_a_pre_patch_sidecar_loads_as_plain(tmp_path):
    # Every existing checkpoint was written by eamax directly and has no `affine` key.
    from eamax.flows import save_conditioner as eamax_save_conditioner

    plain, _ = _pair()
    path = tmp_path / "conditioner"
    eamax_save_conditioner(plain, str(path), context_names=("v", "s", "b"))
    template = make_mlp_conditioner(nnx.Rngs(5), num_in=3, num_mid=16, num_bins=NUM_BINS)
    restored = load_conditioner(template, str(path), context_names=("v", "s", "b"))
    reference = eamax_load_conditioner(template, str(path), context_names=("v", "s", "b"))
    np.testing.assert_array_equal(
        np.asarray(evaluate_pdf_sf(restored, TIMES, CONTEXT)[0]),
        np.asarray(eamax_evaluate_pdf_sf(reference, TIMES, CONTEXT)[0]),
    )
