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
    spline_knots,
    spline_settings,
    input_scaling,
    log_input_scaling,
    flow_options,
    DeepMLP,
    num_hidden_layers,
    train_step,
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


@pytest.mark.parametrize("layout", ["plain", "affine"])
def test_knots_lie_on_the_flow_transform(layout):
    # A knot maps to a knot, so pushing the z knots through the full flow bijector must land
    # exactly on the reported log-time knots — for either layout.
    plain, affine = _pair()
    conditioner = plain if layout == "plain" else affine
    _set_affine_outputs(affine, loc=-1.2, raw_scale=-0.4)
    context = jnp.array([2.0, 1.0, 1.0])

    z_knots, log_t_knots = spline_knots(conditioner, context)
    _, flow = spline_flow(jnp.ones(()), context, conditioner)
    assert z_knots.shape == log_t_knots.shape == (NUM_BINS + 1,)
    np.testing.assert_allclose(
        np.log(np.asarray(flow.bijector.forward(z_knots))), np.asarray(log_t_knots), atol=1e-10,
    )


def _narrow_pair():
    """A default affine conditioner and a narrow/unconstrained one sharing all weights."""
    _, default = _pair()
    narrow = make_mlp_conditioner(
        nnx.Rngs(1), num_in=3, num_mid=16, num_bins=NUM_BINS, affine=True,
        spline_range=3.5, boundary_slopes="unconstrained",
    )
    nnx.update(narrow, nnx.state(default))
    return default, narrow


def test_spline_settings_are_refused_for_a_plain_flow():
    # Without the affine stage the range is absolute log time; narrowing it forbids fast
    # decision times, which is the t0 bias the eamax comment warns about.
    with pytest.raises(ValueError, match="only be changed for an affine flow"):
        make_mlp_conditioner(nnx.Rngs(0), num_in=3, num_bins=NUM_BINS, spline_range=3.5)
    with pytest.raises(ValueError, match="boundary_slopes must be one of"):
        make_mlp_conditioner(nnx.Rngs(0), num_in=3, num_bins=NUM_BINS, affine=True, boundary_slopes="nope")


def test_the_spline_settings_change_the_density():
    default, narrow = _narrow_pair()
    _set_affine_outputs(default, loc=-1.0, raw_scale=-0.3)
    _set_affine_outputs(narrow, loc=-1.0, raw_scale=-0.3)
    assert spline_settings(default) == (5.0, "identity")
    assert spline_settings(narrow) == (3.5, "unconstrained")
    assert not np.allclose(
        np.asarray(evaluate_pdf_sf(default, TIMES, CONTEXT)[0]),
        np.asarray(evaluate_pdf_sf(narrow, TIMES, CONTEXT)[0]),
    )


def test_survival_is_exact_with_unconstrained_boundary_slopes():
    # Beyond the range the spline now extrapolates with learned slopes; it must still be
    # strictly increasing, so S(t) is still the base survival at the inverse.
    _, narrow = _narrow_pair()
    _set_affine_outputs(narrow, loc=-1.0, raw_scale=-0.5)
    # Push the boundary slopes away from 1 so the extrapolation actually differs.
    width = 3 * NUM_BINS + 1
    narrow.linear2.bias.value = narrow.linear2.bias.value.at[2 * NUM_BINS].set(1.5).at[width - 1].set(-1.5)
    context = jnp.array([2.0, 1.0, 1.0])

    grid = jnp.linspace(1e-6, 30.0, 600_000)
    log_pdf, _ = spline_flow(grid, context, narrow)
    cumulative = jnp.cumsum(jnp.exp(log_pdf)) * (grid[1] - grid[0])
    assert float(cumulative[-1]) == pytest.approx(1.0, abs=3e-3)
    for target in (0.1, 0.4, 1.0):
        index = int(jnp.searchsorted(grid, target))
        reported = float(jnp.exp(evaluate_pdf_sf(narrow, grid[index : index + 1], context)[1][0]))
        assert reported == pytest.approx(1.0 - float(cumulative[index]), abs=3e-3)

    z_knots, log_t_knots = spline_knots(narrow, context)
    assert float(z_knots[0]) == pytest.approx(-3.5) and float(z_knots[-1]) == pytest.approx(3.5)
    _, flow = spline_flow(jnp.ones(()), context, narrow)
    np.testing.assert_allclose(
        np.log(np.asarray(flow.bijector.forward(z_knots))), np.asarray(log_t_knots), atol=1e-10,
    )


def test_spline_settings_round_trip_and_mismatches_are_refused(tmp_path):
    default, narrow = _narrow_pair()
    path = tmp_path / "conditioner"
    save_conditioner(narrow, str(path), context_names=("v", "s", "b"))

    from eamax.flows.checkpoint import read_metadata

    metadata = read_metadata(str(path))
    assert (metadata["spline_range"], metadata["boundary_slopes"]) == (3.5, "unconstrained")

    template = make_mlp_conditioner(
        nnx.Rngs(4), num_in=3, num_mid=16, num_bins=NUM_BINS, affine=True,
        spline_range=3.5, boundary_slopes="unconstrained",
    )
    restored = load_conditioner(template, str(path), context_names=("v", "s", "b"))
    assert spline_settings(restored) == (3.5, "unconstrained")
    np.testing.assert_allclose(
        np.asarray(evaluate_pdf_sf(narrow, TIMES, CONTEXT)[0]),
        np.asarray(evaluate_pdf_sf(restored, TIMES, CONTEXT)[0]),
        rtol=1e-6,
    )

    wrong = make_mlp_conditioner(nnx.Rngs(4), num_in=3, num_mid=16, num_bins=NUM_BINS, affine=True)
    with pytest.raises(ValueError, match="flow_spline_range"):
        load_conditioner(wrong, str(path), context_names=("v", "s", "b"))


CRDM_NAMES = ("v", "amp", "tau", "s", "b")
CRDM_BOX = {"v": (0.0, 8.0), "amp": (0.0, 1.0), "tau": (0.0, 0.5), "s": (0.0, 3.0), "b": (0.0, 3.0)}
CRDM_EPS = {"v": 0.05, "amp": 0.01, "tau": 0.005, "s": 0.05, "b": 0.05}


def _scaled_pair(num_in=5, names=CRDM_NAMES, box=CRDM_BOX, eps=CRDM_EPS):
    """An affine conditioner and the same weights with log-scaled inputs."""
    raw = make_mlp_conditioner(nnx.Rngs(3), num_in=num_in, num_mid=16, num_bins=NUM_BINS, affine=True)
    scaling = log_input_scaling(names, eps, box)
    scaled = make_mlp_conditioner(
        nnx.Rngs(7), num_in=num_in, num_mid=16, num_bins=NUM_BINS, affine=True, input_scaling=scaling,
    )
    nnx.update(scaled, nnx.state(raw))
    return raw, scaled, scaling


def test_log_uniform_moments_match_numerical_integration():
    # loc and scale are closed-form moments of log(x + eps) under the uniform box; check them
    # against brute-force quadrature so a sign slip in the antiderivatives cannot hide.
    for low, high, eps in [(0.0, 0.5, 0.005), (0.0, 8.0, 0.05), (0.25, 3.0, 0.0)]:
        x = np.linspace(low, high, 2_000_001)
        y = np.log(x + eps)
        mean, sd = flows_affine._log_uniform_moments(low, high, eps)
        assert mean == pytest.approx(np.trapezoid(y, x) / (high - low), abs=1e-5)
        assert sd == pytest.approx(np.sqrt(np.trapezoid((y - mean) ** 2, x) / (high - low)), abs=1e-5)


def test_log_input_scaling_is_ordered_by_context_and_validated():
    scaling = log_input_scaling(CRDM_NAMES, CRDM_EPS, CRDM_BOX)
    assert scaling["eps"] == tuple(CRDM_EPS[n] for n in CRDM_NAMES)
    assert scaling["loc"][2] == pytest.approx(flows_affine._log_uniform_moments(0.0, 0.5, 0.005)[0])
    with pytest.raises(ValueError, match="must name exactly"):
        log_input_scaling(CRDM_NAMES, {**CRDM_EPS, "t0": 0.1}, CRDM_BOX)
    with pytest.raises(ValueError, match="0 < low \\+ eps"):
        log_input_scaling(CRDM_NAMES, {**CRDM_EPS, "tau": 0.0}, CRDM_BOX)


def test_log_inputs_are_refused_for_a_plain_flow():
    with pytest.raises(ValueError, match="only implemented for an affine flow"):
        make_mlp_conditioner(
            nnx.Rngs(0), num_in=5, num_bins=NUM_BINS,
            input_scaling=log_input_scaling(CRDM_NAMES, CRDM_EPS, CRDM_BOX),
        )


def test_scaling_off_leaves_the_flow_unchanged():
    raw, _, _ = _scaled_pair()
    assert input_scaling(raw) is None
    context = jnp.array([[4.0, 0.3, 0.1, 0.8, 0.9]])
    np.testing.assert_array_equal(
        np.asarray(flows_affine._conditioner_outputs(raw, context)), np.asarray(raw(context))
    )


def test_scaled_flow_is_the_raw_flow_on_transformed_inputs():
    # The only thing scaling may change is what the network sees: the scaled conditioner on x
    # must be exactly the unscaled one on (log(x + eps) - loc) / scale.
    raw, scaled, scaling = _scaled_pair()
    context = jnp.array([[[4.0, 0.3, 0.1, 0.8, 0.9]], [[1.2, 0.0, 0.05, 1.0, 1.4]]])
    times = jnp.tile(jnp.linspace(0.03, 1.5, 30), (2, 1))
    transformed = (
        jnp.log(context + jnp.asarray(scaling["eps"])) - jnp.asarray(scaling["loc"])
    ) / jnp.asarray(scaling["scale"])
    for a, b in zip(
        evaluate_pdf_sf(scaled, times, context), evaluate_pdf_sf(raw, times, transformed), strict=True,
    ):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=1e-12)


def test_zero_amp_and_tau_stay_finite_through_the_clamp():
    # amp = 0 is a plain Wald and tau -> 0 is the edge of the box; with the clamp at 0 and eps
    # inside the log, both must give finite densities and gradients.
    _, scaled, _ = _scaled_pair()
    from confrdm_jax.likelihoods import _flow_accumulator

    accumulator = _flow_accumulator(scaled, CRDM_NAMES, context_bounds=CRDM_BOX)
    t = jnp.linspace(0.05, 1.0, 20)

    def total(values):
        params = {name: jnp.full(t.shape, values[i]) for i, name in enumerate(CRDM_NAMES)}
        log_pdf, log_sf = accumulator.log_pdf_sf(t, params)
        return jnp.sum(log_pdf) + jnp.sum(log_sf)

    for values in (jnp.array([4.0, 0.0, 0.0, 0.8, 0.9]), jnp.array([4.0, -0.2, -0.1, 0.8, 0.9])):
        value, grad = jax.value_and_grad(total)(values)
        assert np.isfinite(float(value))
        assert np.all(np.isfinite(np.asarray(grad)))


def test_knots_lie_on_the_scaled_flow_transform():
    _, scaled, _ = _scaled_pair()
    _set_affine_outputs(scaled, loc=-1.2, raw_scale=-0.4)
    context = jnp.array([4.0, 0.3, 0.1, 0.8, 0.9])
    z_knots, log_t_knots = spline_knots(scaled, context)
    _, flow = spline_flow(jnp.ones(()), context, scaled)
    np.testing.assert_allclose(
        np.log(np.asarray(flow.bijector.forward(z_knots))), np.asarray(log_t_knots), atol=1e-10,
    )


def test_scaling_round_trips_and_mismatches_are_refused(tmp_path):
    raw, scaled, scaling = _scaled_pair()
    names = list(CRDM_NAMES)
    scaled_path, raw_path = tmp_path / "scaled", tmp_path / "raw"
    save_conditioner(scaled, str(scaled_path), context_names=names)
    save_conditioner(raw, str(raw_path), context_names=names)

    from eamax.flows.checkpoint import read_metadata

    assert read_metadata(str(scaled_path))["input_scaling"]["loc"] == list(scaling["loc"])
    assert read_metadata(str(raw_path))["input_scaling"] is None

    def template(with_scaling):
        return make_mlp_conditioner(
            nnx.Rngs(11), num_in=5, num_mid=16, num_bins=NUM_BINS, affine=True,
            input_scaling=scaling if with_scaling else None,
        )

    restored = load_conditioner(template(True), str(scaled_path), context_names=names)
    assert input_scaling(restored) == scaling
    context = jnp.array([[[4.0, 0.3, 0.1, 0.8, 0.9]]])
    times = jnp.linspace(0.05, 1.0, 10)[None]
    np.testing.assert_allclose(
        np.asarray(evaluate_pdf_sf(scaled, times, context)[0]),
        np.asarray(evaluate_pdf_sf(restored, times, context)[0]),
        rtol=1e-6,
    )

    with pytest.raises(ValueError, match="flow_log_inputs"):
        load_conditioner(template(False), str(scaled_path), context_names=names)
    with pytest.raises(ValueError, match="flow_log_inputs"):
        load_conditioner(template(True), str(raw_path), context_names=names)
    shifted = {**scaling, "loc": tuple(x + 0.1 for x in scaling["loc"])}
    wrong_box = make_mlp_conditioner(
        nnx.Rngs(11), num_in=5, num_mid=16, num_bins=NUM_BINS, affine=True, input_scaling=shifted,
    )
    with pytest.raises(ValueError, match="flow_log_inputs"):
        load_conditioner(wrong_box, str(scaled_path), context_names=names)


@pytest.mark.parametrize("model", ["rdm", "crdm"])
def test_config_log_inputs_follow_the_training_box(model):
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    config_dir = str(Path(__file__).parents[1] / "conf_jax")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        off = compose("config", overrides=[f"model={model}"])
        on = compose("config", overrides=[
            f"model={model}", "model.flow_affine=true", "model.flow_log_inputs=true",
            "model.training_prior.b_max=2.5",
        ])

    assert flow_options(off.model)["input_scaling"] is None
    options = flow_options(on.model)
    names = list(on.model.context_names)
    b = names.index("b")
    b_min = float(on.model.flow_context_bounds["b"][0])
    expected = flows_affine._log_uniform_moments(b_min, 2.5, float(on.model.flow_input_log_eps["b"]))
    assert options["input_scaling"]["loc"][b] == pytest.approx(expected[0])
    assert options["input_scaling"]["scale"][b] == pytest.approx(expected[1])
    conditioner = make_mlp_conditioner(nnx.Rngs(0), num_in=len(names), num_bins=4, **options)
    assert input_scaling(conditioner) == options["input_scaling"]


def test_one_hidden_layer_is_eamax_mlp():
    # The default must not change any existing checkpoint's structure.
    from eamax.flows.model import MLP

    for affine in (False, True):
        conditioner = make_mlp_conditioner(nnx.Rngs(0), num_in=5, num_mid=16, num_bins=NUM_BINS, affine=affine)
        assert type(conditioner) is MLP
        assert num_hidden_layers(conditioner) == 1
    with pytest.raises(ValueError, match="at least 1"):
        make_mlp_conditioner(nnx.Rngs(0), num_in=5, num_bins=NUM_BINS, affine=True, num_hidden=0)


@pytest.mark.parametrize("affine", [False, True])
def test_deep_conditioner_is_the_composed_layers(affine):
    conditioner = make_mlp_conditioner(
        nnx.Rngs(0), num_in=5, num_mid=16, num_bins=NUM_BINS, affine=affine, num_hidden=3,
    )
    assert isinstance(conditioner, DeepMLP) and num_hidden_layers(conditioner) == 3
    assert conditioner_layout(conditioner) == (NUM_BINS, affine)

    x = jnp.array([[4.0, 0.3, 0.1, 0.8, 0.9]])
    h = nnx.gelu(conditioner.linear1(x))
    h = nnx.gelu(conditioner.linear_mid1(h))
    h = nnx.gelu(conditioner.linear_mid2(h))
    np.testing.assert_allclose(np.asarray(conditioner(x)), np.asarray(conditioner.linear2(h)), rtol=1e-12)


def test_deep_log_scaled_affine_flow_trains():
    # The full experimental stack -- two hidden layers, affine stage, log inputs -- must give a
    # finite loss, reach every layer with its gradient, and survive a jitted train step.
    import optax

    conditioner = make_mlp_conditioner(
        nnx.Rngs(5), num_in=5, num_mid=16, num_bins=NUM_BINS, affine=True, num_hidden=2,
        input_scaling=log_input_scaling(CRDM_NAMES, CRDM_EPS, CRDM_BOX),
    )
    context = jnp.tile(jnp.array([[[4.0, 0.3, 0.1, 0.8, 0.9]]]), (4, 1, 1))
    data = jnp.tile(jnp.linspace(0.05, 1.0, 16), (4, 1)).at[0, -1].set(jnp.inf)

    grads = jax.grad(lambda c: loss_fn(c, data, context, 4.0))(conditioner)
    for layer in ("linear1", "linear_mid1", "linear2"):
        kernel = np.asarray(getattr(grads, layer).kernel.value)
        assert np.all(np.isfinite(kernel)) and np.any(kernel != 0.0)

    optimizer = nnx.Optimizer(conditioner, optax.adam(1e-3), wrt=nnx.Param)
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
    train_step(conditioner, optimizer, metrics, data, context, 4.0)
    assert np.isfinite(float(metrics.compute()["loss"]))


def test_depth_round_trips_and_mismatches_are_refused(tmp_path):
    deep = make_mlp_conditioner(nnx.Rngs(1), num_in=3, num_mid=16, num_bins=NUM_BINS, affine=True, num_hidden=2)
    shallow = make_mlp_conditioner(nnx.Rngs(1), num_in=3, num_mid=16, num_bins=NUM_BINS, affine=True)
    deep_path, shallow_path = tmp_path / "deep", tmp_path / "shallow"
    save_conditioner(deep, str(deep_path), context_names=("v", "s", "b"))
    save_conditioner(shallow, str(shallow_path), context_names=("v", "s", "b"))

    from eamax.flows.checkpoint import read_metadata

    assert read_metadata(str(deep_path))["num_hidden"] == 2
    assert read_metadata(str(shallow_path))["num_hidden"] == 1

    template = make_mlp_conditioner(nnx.Rngs(9), num_in=3, num_mid=16, num_bins=NUM_BINS, affine=True, num_hidden=2)
    restored = load_conditioner(template, str(deep_path), context_names=("v", "s", "b"))
    np.testing.assert_allclose(
        np.asarray(evaluate_pdf_sf(deep, TIMES, CONTEXT)[0]),
        np.asarray(evaluate_pdf_sf(restored, TIMES, CONTEXT)[0]),
        rtol=1e-6,
    )
    with pytest.raises(ValueError, match="flow_num_hidden"):
        load_conditioner(shallow, str(deep_path), context_names=("v", "s", "b"))
    with pytest.raises(ValueError, match="flow_num_hidden"):
        load_conditioner(template, str(shallow_path), context_names=("v", "s", "b"))


def test_config_selects_depth():
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    config_dir = str(Path(__file__).parents[1] / "conf_jax")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        default = compose("config", overrides=["model=crdm"])
        deep = compose("config", overrides=["model=crdm", "model.flow_affine=true", "model.flow_num_hidden=2"])
    assert flow_options(default.model)["num_hidden"] == 1
    options = flow_options(deep.model)
    assert options["num_hidden"] == 2
    conditioner = make_mlp_conditioner(nnx.Rngs(0), num_in=5, num_bins=4, **options)
    assert isinstance(conditioner, DeepMLP)
