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
"""

import json
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
from eamax.flows.model import MLP, RANGE_MAX, RANGE_MIN
from flax import nnx
from jax.scipy import stats

__all__ = [
    "MIN_SCALE",
    "FlowAccumulator",
    "conditioner_layout",
    "evaluate_pdf_sf",
    "load_conditioner",
    "loss_fn",
    "make_mlp_conditioner",
    "save_conditioner",
    "spline_flow",
    "train_step",
]

#: Floor on the per-context scale, so the affine stage can never collapse to a point mass.
MIN_SCALE = 1e-3

# softplus(_SCALE_OFFSET) + MIN_SCALE == 1, so a zero raw output is the identity.
_SCALE_OFFSET = float(jnp.log(jnp.expm1(1.0 - MIN_SCALE)))


def make_mlp_conditioner(rngs, num_in, num_mid=64, num_bins=4, affine=False):
    """`eamax.flows.make_mlp_conditioner`, optionally with the two affine outputs.

    Parameters
    ----------
    rngs : flax.nnx.Rngs
    num_in, num_mid, num_bins : int
        As in `eamax`.
    affine : bool, optional
        Append a per-context ``loc`` and raw ``scale`` on log decision time. ``False``
        returns exactly the `eamax` conditioner.

    Returns
    -------
    eamax.flows.MLP
    """
    if not affine:
        return _eamax_make_mlp_conditioner(rngs, num_in=num_in, num_mid=num_mid, num_bins=num_bins)
    return MLP(din=num_in, dmid=num_mid, dout=3 * num_bins + 3, rngs=rngs)


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


def spline_flow(data, context, conditioner):
    """`eamax.flows.spline_flow`, with the affine stage when the conditioner has one.

    A plain conditioner is handed straight to `eamax`, so its densities are bit-identical
    to what they were before this module existed.
    """
    params = conditioner(context)
    num_bins, affine = _layout_from_width(params.shape[-1])
    if not affine:
        return _eamax_spline_flow(data, context, conditioner)

    spline = distrax.RationalQuadraticSpline(
        params[..., : 3 * num_bins + 1],
        range_min=RANGE_MIN,
        range_max=RANGE_MAX,
        boundary_slopes="identity",
        min_bin_size=1e-4,
    )
    loc = params[..., -2]
    scale = jax.nn.softplus(params[..., -1] + _SCALE_OFFSET) + MIN_SCALE
    flow = distrax.Transformed(
        distrax.Normal(loc=0.0, scale=1.0),
        distrax.Chain([_exponential(), distrax.ScalarAffine(shift=loc, scale=scale), spline]),
    )
    return flow.log_prob(data), flow


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
        (Path(path) / SIDECAR_NAME).write_text(json.dumps(metadata, indent=2))


def load_conditioner(conditioner, path, step=0, context_names=None):
    """`eamax.flows.load_conditioner`, refusing a template with the wrong layout.

    Orbax would fail on the shape mismatch anyway, but with an error that says nothing about
    ``model.flow_affine``. A sidecar without the flag predates this module and is plain.
    """
    metadata = read_metadata(path)
    if metadata is not None:
        recorded = bool(metadata.get("affine", False))
        expected = conditioner_layout(conditioner)[1]
        if recorded != expected:
            raise ValueError(
                f"Checkpoint at {path} was trained with affine={recorded}, but the conditioner "
                f"it is being loaded into has affine={expected}. Set model.flow_affine to match."
            )
    return _eamax_load_conditioner(conditioner, path, step=step, context_names=context_names)
