"""Local patch over `eamax.flows`: an optional per-context location and scale on log time.

**Experimental.** This lives here rather than in `eamax` so it can be tried without
touching the shared library. If it earns its place it belongs in `eamax.flows.model`; until
then, everything that builds, trains, saves, loads or evaluates a conditioner in this
repository goes through this module, which falls back to `eamax` unchanged for a plain
conditioner.

**Why.** The `eamax` flow is ``Z ~ N(0, 1) -> spline -> exp -> T`` with the spline active
on a fixed ``[-5, 5]`` on *both* axes. Every context has to share that one window of log
decision time, but a single context only occupies ~3 units of it while the medians of
different contexts spread over 6-9 (measured over the CRDM training box). The spline
therefore spends most of its bins reaching the corners, and on the trained CRDM conditioner
two of twelve bins carried 98.6% of the base mass — visible as kinks at the knots and as a
~10% density error at the peak.

**What.** With ``affine=True`` the conditioner emits two more outputs and the flow becomes

    Z ~ N(0, 1)  --spline-->  Y  --loc(ctx) + scale(ctx) * Y-->  log T  --exp-->  T

so the spline models only the *shape* of a standardised log decision time and its bins
land where that context's mass is. Outside the spline's range the transform is the identity
in the standardised coordinate, so the tails become log-normal with the context's own scale
rather than being pinned to ``exp(+-5)``. Every stage is strictly increasing (``scale > 0``),
so the survival function stays exact given the flow.

**Layout.** Output width ``3K + 1`` is a plain `eamax` conditioner; ``3K + 3`` is the
affine one, with ``[loc, raw_scale]`` appended *after* the spline parameters so anything
reading the first ``2K`` entries as bin widths and heights still does. The two widths never
coincide (they differ mod 3), so the layout is read off the weights, and a checkpoint
cannot be evaluated with the wrong one. ``scale = softplus(raw + c) + MIN_SCALE`` with ``c``
chosen so ``raw = 0`` gives exactly 1: zeroed affine outputs reproduce the plain flow.

**Spline settings (affine only).** `eamax` fixes the spline to ``[-5, 5]`` with slope 1 at
both ends (``boundary_slopes="identity"``). Both corners are pinned, and a skewed
decision-time distribution cannot meet them with one shift and scale, so the trained
conditioners spend ~8 of 12 bins bending into the tails. An affine flow may instead use a
narrower ``spline_range`` and ``boundary_slopes="unconstrained"``: the range is then
relative to each context, and beyond it the tails continue linearly in the standardised
coordinate with *learned* slopes, so lower and upper tails can differ. For a plain flow the
range is absolute log time, where narrowing it forbids fast decision times, so it is refused.
The settings are plain attributes on the conditioner (part of its graph, not its weights),
recorded in the sidecar and checked on load: the same weights under a different range are a
different density.

**Log-scaled inputs (affine only).** The conditioner can see each input as
``u = (log(x + eps) - loc) / scale`` instead of ``x``. The diffusion's two exact symmetries
are multiplicative on the raw scale -- rescaling time maps ``(v, amp, tau, s, b)`` to
``(c v, amp, tau / c, sqrt(c) s, b)`` and divides decision times by ``c``; rescaling space
maps it to ``(a v, a amp, tau, a s, a b)`` and changes nothing -- and both become shifts in
log space, where the affine stage's location is linear along them. A log scale also spaces
tau's relative changes evenly, which is where the trained flows are weakest. ``eps`` keeps
zero inputs finite (``amp = 0`` is a plain Wald); ``loc`` and ``scale`` are the mean and SD of
``log(x + eps)`` under the uniform training box, in closed form, so they are fixed by the
config rather than learned. The transform is applied in :func:`_conditioner_outputs`, the one
place every context meets the network -- the training loss and the likelihood both -- and,
like the spline settings, it is stored on the conditioner, recorded in the sidecar and checked
on load.

**Conditioner depth.** `eamax`'s conditioner has one hidden layer. ``num_hidden > 1`` builds a
:class:`DeepMLP` with further ``dmid -> dmid`` GELU layers between ``linear1`` and ``linear2``,
so the output layer keeps its name and :func:`conditioner_layout` still reads it. One hidden
layer returns `eamax`'s MLP itself, so existing checkpoints and their structure are untouched.
The depth changes the checkpoint's structure; it is recorded in the sidecar so that a mismatch
fails with a message naming ``model.flow_num_hidden`` rather than an Orbax tree error.
"""

import json
import math
from pathlib import Path

import distrax
import jax
import jax.numpy as jnp
from eamax.flows import FlowAccumulator as _EamaxFlowAccumulator
from eamax.flows import load_conditioner as _eamax_load_conditioner
from eamax.flows import make_mlp_conditioner as _eamax_make_mlp_conditioner
from eamax.flows import save_conditioner as _eamax_save_conditioner
from eamax.flows import spline_flow as _eamax_spline_flow
from eamax.flows.checkpoint import SIDECAR_NAME, read_metadata
from eamax.flows.model import MLP, RANGE_MAX
from flax import nnx
from jax.scipy import stats

__all__ = [
    "MIN_SCALE",
    "FlowAccumulator",
    "DeepMLP",
    "conditioner_layout",
    "evaluate_pdf_sf",
    "flow_options",
    "input_scaling",
    "load_conditioner",
    "log_input_scaling",
    "num_hidden_layers",
    "loss_fn",
    "make_mlp_conditioner",
    "save_conditioner",
    "spline_flow",
    "spline_knots",
    "spline_settings",
    "train_step",
]

#: `eamax`'s spline settings, and the defaults here.
DEFAULT_SPLINE_RANGE = RANGE_MAX
DEFAULT_BOUNDARY_SLOPES = "identity"
_BOUNDARY_SLOPES = ("identity", "unconstrained", "lower_identity", "upper_identity")

#: Floor on the per-context scale, so the affine stage can never collapse to a point mass.
MIN_SCALE = 1e-3

# softplus(_SCALE_OFFSET) + MIN_SCALE == 1, so a zero raw output is the identity.
_SCALE_OFFSET = float(jnp.log(jnp.expm1(1.0 - MIN_SCALE)))


class DeepMLP(MLP):
    """`eamax.flows.MLP` with ``num_hidden - 1`` extra ``dmid -> dmid`` GELU layers.

    ``linear1`` and ``linear2`` keep their roles and names -- input and output layer -- and the
    extra layers are ``linear_mid1``, ``linear_mid2``, ... in order.
    """

    def __init__(self, din: int, dmid: int, dout: int, num_hidden: int, *, rngs: nnx.Rngs):
        super().__init__(din, dmid, dout, rngs=rngs)
        self.num_hidden = int(num_hidden)
        for index in range(1, self.num_hidden):
            setattr(self, f"linear_mid{index}", nnx.Linear(dmid, dmid, rngs=rngs))

    def __call__(self, x):
        x = nnx.gelu(self.linear1(x))
        for index in range(1, self.num_hidden):
            x = nnx.gelu(getattr(self, f"linear_mid{index}")(x))
        return self.linear2(x)


def num_hidden_layers(conditioner):
    """Hidden layers in a conditioner: 1 for `eamax`'s MLP."""
    return int(getattr(conditioner, "num_hidden", 1))


def make_mlp_conditioner(
    rngs, num_in, num_mid=64, num_bins=4, affine=False,
    spline_range=DEFAULT_SPLINE_RANGE, boundary_slopes=DEFAULT_BOUNDARY_SLOPES,
    input_scaling=None, num_hidden=1,
):
    """`eamax.flows.make_mlp_conditioner`, optionally with the two affine outputs.

    Parameters
    ----------
    rngs : flax.nnx.Rngs
    num_in, num_mid, num_bins : int
        As in `eamax`.
    affine : bool, optional
        Append a per-context ``loc`` and raw ``scale`` on log decision time. ``False``
        returns exactly the `eamax` conditioner.
    spline_range : float, optional
        The spline is active on ``[-spline_range, spline_range]``. Affine only.
    boundary_slopes : str, optional
        `distrax` boundary condition; ``"unconstrained"`` learns both end slopes. Affine only.
    input_scaling : dict, optional
        ``{"eps", "loc", "scale"}``, each a length-``num_in`` sequence in context order, as
        returned by :func:`log_input_scaling`. Affine only.
    num_hidden : int, optional
        Hidden layers in the conditioner. ``1`` is `eamax`'s MLP; more gives a
        :class:`DeepMLP`.

    Returns
    -------
    eamax.flows.MLP
    """
    spline_range = float(spline_range)
    boundary_slopes = str(boundary_slopes)
    num_hidden = int(num_hidden)
    if num_hidden < 1:
        raise ValueError(f"num_hidden must be at least 1, got {num_hidden}.")

    def build(dout):
        if num_hidden == 1:
            return MLP(din=num_in, dmid=num_mid, dout=dout, rngs=rngs)
        return DeepMLP(din=num_in, dmid=num_mid, dout=dout, num_hidden=num_hidden, rngs=rngs)

    if boundary_slopes not in _BOUNDARY_SLOPES:
        raise ValueError(f"boundary_slopes must be one of {_BOUNDARY_SLOPES}, got {boundary_slopes!r}.")
    if not spline_range > 0.0:
        raise ValueError(f"spline_range must be positive, got {spline_range}.")
    if not affine:
        if (spline_range, boundary_slopes) != (DEFAULT_SPLINE_RANGE, DEFAULT_BOUNDARY_SLOPES):
            raise ValueError(
                "spline_range and boundary_slopes can only be changed for an affine flow: without "
                "the per-context location and scale the range is absolute log time, and "
                "narrowing it forbids fast decision times."
            )
        if input_scaling is not None:
            raise ValueError(
                "Log-scaled inputs are only implemented for an affine flow (model.flow_affine=true)."
            )
        if num_hidden == 1:
            return _eamax_make_mlp_conditioner(rngs, num_in=num_in, num_mid=num_mid, num_bins=num_bins)
        return build(3 * num_bins + 1)
    conditioner = build(3 * num_bins + 3)
    conditioner.spline_range = spline_range
    conditioner.boundary_slopes = boundary_slopes
    if input_scaling is not None:
        columns = {key: tuple(float(x) for x in input_scaling[key]) for key in ("eps", "loc", "scale")}
        if any(len(column) != num_in for column in columns.values()):
            raise ValueError(f"input_scaling needs {num_in} entries per key, got {columns}.")
        if not all(sd > 0.0 for sd in columns["scale"]):
            raise ValueError(f"input_scaling scales must be positive, got {columns['scale']}.")
        conditioner.input_log_eps = columns["eps"]
        conditioner.input_loc = columns["loc"]
        conditioner.input_scale = columns["scale"]
    return conditioner


def input_scaling(conditioner):
    """The conditioner's log-input scaling as ``{"eps", "loc", "scale"}`` tuples, or ``None``."""
    if getattr(conditioner, "input_log_eps", None) is None:
        return None
    return {
        "eps": tuple(conditioner.input_log_eps),
        "loc": tuple(conditioner.input_loc),
        "scale": tuple(conditioner.input_scale),
    }


def _log_uniform_moments(low, high, eps):
    """Mean and SD of ``log(x + eps)`` for ``x ~ Uniform(low, high)``, in closed form.

    With ``y = x + eps`` uniform on ``[a, b]``: ``E[log y] = [F]_a^b / (b - a)`` with
    ``F(y) = y log y - y``, and ``E[log^2 y] = [G]_a^b / (b - a)`` with
    ``G(y) = y (log^2 y - 2 log y + 2)``.
    """
    a, b = low + eps, high + eps
    if not (a > 0.0 and b > a):
        raise ValueError(f"log input scaling needs 0 < low + eps < high + eps, got low={low}, high={high}, eps={eps}.")
    F = lambda y: y * math.log(y) - y  # noqa: E731
    G = lambda y: y * (math.log(y) ** 2 - 2.0 * math.log(y) + 2.0)  # noqa: E731
    mean = (F(b) - F(a)) / (b - a)
    var = (G(b) - G(a)) / (b - a) - mean**2
    return mean, math.sqrt(max(var, 0.0))


def log_input_scaling(context_names, eps, bounds):
    """Fixed log-input scaling for a conditioner trained on a uniform box.

    Parameters
    ----------
    context_names : sequence of str
        The flow's inputs, in training order.
    eps : mapping of str to float
        Offset added before the log, per input. Must name exactly ``context_names``.
    bounds : mapping of str to (low, high)
        The uniform training box per input -- ``model.flow_context_bounds``.

    Returns
    -------
    dict
        ``{"eps", "loc", "scale"}``, tuples in ``context_names`` order.
    """
    names = [str(name) for name in context_names]
    eps = {str(k): float(v) for k, v in eps.items()}
    if set(eps) != set(names):
        raise ValueError(f"flow_input_log_eps must name exactly {names}, got {sorted(eps)}.")
    columns = {"eps": [], "loc": [], "scale": []}
    for name in names:
        low, high = (float(limit) for limit in bounds[name])
        mean, sd = _log_uniform_moments(low, high, eps[name])
        columns["eps"].append(eps[name]); columns["loc"].append(mean); columns["scale"].append(sd)
    return {key: tuple(value) for key, value in columns.items()}


def flow_options(model_cfg):
    """The conditioner keyword arguments a model config selects, for the scripts.

    One place for the ``flow_*`` keys, so training and both recovery scripts build the same
    conditioner template from the same config.
    """
    options = {
        "affine": bool(model_cfg["flow_affine"]),
        "spline_range": float(model_cfg["flow_spline_range"]),
        "boundary_slopes": str(model_cfg["flow_boundary_slopes"]),
        "input_scaling": None,
        "num_hidden": int(model_cfg["flow_num_hidden"]),
    }
    if bool(model_cfg["flow_log_inputs"]):
        options["input_scaling"] = log_input_scaling(
            model_cfg["context_names"], model_cfg["flow_input_log_eps"], model_cfg["flow_context_bounds"],
        )
    return options


def _conditioner_outputs(conditioner, context):
    """The conditioner applied to `context`, through its log-input scaling if it has one.

    Every path from a context to spline parameters goes through here, so training and
    inference cannot see differently scaled inputs.
    """
    scaling = input_scaling(conditioner)
    if scaling is None:
        return conditioner(context)
    context = jnp.asarray(context)
    eps, loc, scale = (jnp.asarray(scaling[key], dtype=context.dtype) for key in ("eps", "loc", "scale"))
    return conditioner((jnp.log(context + eps) - loc) / scale)


def spline_settings(conditioner):
    """``(spline_range, boundary_slopes)`` of a conditioner; `eamax`'s for one without them."""
    return (
        float(getattr(conditioner, "spline_range", DEFAULT_SPLINE_RANGE)),
        str(getattr(conditioner, "boundary_slopes", DEFAULT_BOUNDARY_SLOPES)),
    )


def _layout_from_width(width):
    if width % 3 == 1:
        return (width - 1) // 3, False
    if width % 3 == 0:
        return (width - 3) // 3, True
    raise ValueError(
        f"Conditioner output width {width} is neither 3K+1 (plain) nor 3K+3 (affine)."
    )


def conditioner_layout(conditioner):
    """``(num_bins, affine)`` of a conditioner, read off its output layer."""
    return _layout_from_width(int(conditioner.linear2.out_features))


def _exponential():
    return distrax.Lambda(
        forward=jnp.exp,
        inverse=jnp.log,
        forward_log_det_jacobian=lambda z: z,
        inverse_log_det_jacobian=lambda x: -jnp.log(x),
        event_ndims_in=0,
        event_ndims_out=0,
    )


def _spline(spline_params, spline_range=DEFAULT_SPLINE_RANGE, boundary_slopes=DEFAULT_BOUNDARY_SLOPES):
    """The rational-quadratic spline; with the defaults, exactly as `eamax.flows.spline_flow`."""
    return distrax.RationalQuadraticSpline(
        spline_params,
        range_min=-spline_range,
        range_max=spline_range,
        boundary_slopes=boundary_slopes,
        min_bin_size=1e-4,
    )


def _loc_scale(params):
    return params[..., -2], jax.nn.softplus(params[..., -1] + _SCALE_OFFSET) + MIN_SCALE


def spline_flow(data, context, conditioner):
    """`eamax.flows.spline_flow`, with the affine stage when the conditioner has one.

    A plain conditioner is handed straight to `eamax`, so its densities are bit-identical
    to what they were before this module existed.
    """
    params = _conditioner_outputs(conditioner, context)
    num_bins, affine = _layout_from_width(params.shape[-1])
    if not affine:
        return _eamax_spline_flow(data, context, conditioner)

    spline = _spline(params[..., : 3 * num_bins + 1], *spline_settings(conditioner))
    loc, scale = _loc_scale(params)
    flow = distrax.Transformed(
        distrax.Normal(loc=0.0, scale=1.0),
        distrax.Chain([_exponential(), distrax.ScalarAffine(shift=loc, scale=scale), spline]),
    )
    return flow.log_prob(data), flow


def spline_knots(conditioner, context):
    """The spline's knots for one context: base ``z`` and the log decision time they map to.

    For figures. The knots are where the transform's curvature, and so the density's slope,
    can jump. A plain flow maps ``z`` knots to ``log t = y`` knots directly; an affine flow
    to ``log t = loc + scale * y``, so its knots move with the context rather than sitting
    at fixed log times. Bin sizes include the spline's ``min_bin_size``, as `distrax` builds
    them.

    Parameters
    ----------
    conditioner : eamax.flows.MLP
    context : array
        Shape ``(num_in,)``.

    Returns
    -------
    z_knots, log_t_knots : array
        Shape ``(num_bins + 1,)`` each.
    """
    params = _conditioner_outputs(conditioner, jnp.asarray(context))
    num_bins, affine = _layout_from_width(params.shape[-1])
    spline = _spline(params[..., : 3 * num_bins + 1], *spline_settings(conditioner))
    log_t = spline.y_pos
    if affine:
        loc, scale = _loc_scale(params)
        log_t = loc + scale * log_t
    return spline.x_pos, log_t


def evaluate_pdf_sf(conditioner, data, context):
    """`eamax.flows.evaluate_pdf_sf` over :func:`spline_flow`."""
    conditioner.eval()
    _, flow = spline_flow(jnp.squeeze(data), context, conditioner)
    return flow.log_prob(data), stats.norm.logsf(flow.bijector.inverse(data))


def loss_fn(conditioner, data, context, t_max=None):
    """`eamax.flows.loss_fn` (right-censored NLL) over :func:`spline_flow`."""
    flat = jnp.squeeze(data)
    is_valid = jnp.isfinite(flat) & (flat > 0.0)
    safe = jnp.where(is_valid, flat, 1.0 if t_max is None else t_max)

    log_pdf, flow = spline_flow(safe, context, conditioner)
    log_sf = stats.norm.logsf(flow.bijector.inverse(safe))

    return -jnp.mean(jnp.where(is_valid, log_pdf, log_sf))


@nnx.jit(static_argnames="t_max")
def train_step(conditioner, optimizer, metrics, data, context, t_max=None):
    """`eamax.flows.train_step` over :func:`loss_fn`."""
    loss, grads = nnx.value_and_grad(loss_fn)(conditioner, data, context, t_max)
    metrics.update(loss=loss)
    optimizer.update(conditioner, grads)


class FlowAccumulator(_EamaxFlowAccumulator):
    """`eamax.flows.FlowAccumulator` evaluated through :func:`spline_flow`."""

    def __init__(self, conditioner, context_names, transform=None, dtype=jnp.float32, remat=False):
        super().__init__(conditioner, context_names, transform=transform, dtype=dtype, remat=remat)

        def forward(data, context):
            return evaluate_pdf_sf(self.conditioner, data, context)

        self._forward = jax.checkpoint(forward) if remat else forward

    def sample(self, key, params):
        context = self.build_context(params)
        _, flow = spline_flow(
            jnp.ones(jnp.shape(context)[:-1], dtype=self.dtype), context, self.conditioner
        )
        return flow.sample(seed=key)


def save_conditioner(conditioner, path, step=0, context_names=None, num_mid=None, num_bins=None):
    """`eamax.flows.save_conditioner`, recording the layout's ``affine`` flag in the sidecar."""
    _eamax_save_conditioner(
        conditioner, path, step=step, context_names=context_names,
        num_mid=num_mid, num_bins=num_bins,
    )
    metadata = read_metadata(path)
    if metadata is not None:
        metadata["affine"] = conditioner_layout(conditioner)[1]
        metadata["spline_range"], metadata["boundary_slopes"] = spline_settings(conditioner)
        scaling = input_scaling(conditioner)
        metadata["input_scaling"] = None if scaling is None else {k: list(v) for k, v in scaling.items()}
        metadata["num_hidden"] = num_hidden_layers(conditioner)
        (Path(path) / SIDECAR_NAME).write_text(json.dumps(metadata, indent=2))


def _same_scaling(recorded, expected):
    if recorded is None or expected is None:
        return recorded is None and expected is None
    return all(
        len(recorded[key]) == len(expected[key])
        and all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15) for a, b in zip(recorded[key], expected[key], strict=True))
        for key in ("eps", "loc", "scale")
    )


def load_conditioner(conditioner, path, step=0, context_names=None):
    """`eamax.flows.load_conditioner`, refusing a template with the wrong layout or spline.

    Orbax would fail on an affine mismatch anyway, but with an error that says nothing about
    ``model.flow_affine``; a spline-setting mismatch it would not catch at all. A sidecar
    without these keys predates this module and is plain `eamax`.
    """
    metadata = read_metadata(path)
    if metadata is not None:
        recorded_depth = int(metadata.get("num_hidden", 1))
        if recorded_depth != num_hidden_layers(conditioner):
            raise ValueError(
                f"Checkpoint at {path} was trained with num_hidden={recorded_depth}, but the "
                f"conditioner it is being loaded into has {num_hidden_layers(conditioner)}. Set "
                "model.flow_num_hidden to match."
            )
        recorded = bool(metadata.get("affine", False))
        expected = conditioner_layout(conditioner)[1]
        if recorded != expected:
            raise ValueError(
                f"Checkpoint at {path} was trained with affine={recorded}, but the conditioner "
                f"it is being loaded into has affine={expected}. Set model.flow_affine to match."
            )
        recorded_settings = (
            float(metadata.get("spline_range", DEFAULT_SPLINE_RANGE)),
            str(metadata.get("boundary_slopes", DEFAULT_BOUNDARY_SLOPES)),
        )
        if recorded_settings != spline_settings(conditioner):
            raise ValueError(
                f"Checkpoint at {path} was trained with (spline_range, boundary_slopes) = "
                f"{recorded_settings}, but the conditioner it is being loaded into has "
                f"{spline_settings(conditioner)}. Set model.flow_spline_range and "
                "model.flow_boundary_slopes to match."
            )
        recorded_scaling = metadata.get("input_scaling")
        expected_scaling = input_scaling(conditioner)
        if not _same_scaling(recorded_scaling, expected_scaling):
            raise ValueError(
                f"Checkpoint at {path} was trained with input_scaling={recorded_scaling}, but the "
                f"conditioner it is being loaded into has {expected_scaling}. Set "
                "model.flow_log_inputs (and model.flow_input_log_eps, and the training box) to match."
            )
    return _eamax_load_conditioner(conditioner, path, step=step, context_names=context_names)
