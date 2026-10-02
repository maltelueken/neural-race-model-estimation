"""Export a trained spline flow (conditioner + bijector) to ONNX via jax2onnx.

Pipeline: Orbax checkpoint -> JAX function (data, context) -> (log_pdf, log_sf)
       -> jax2onnx.to_onnx -> .onnx

The conditioner template is rebuilt from the checkpoint's sidecar (depth, affine layout,
spline settings, log-input scaling, width, bins), so plain and affine / log-input / deep
flows all export, and a checkpoint cannot be loaded into the wrong architecture. The exported
graph evaluates the same flow as :func:`confrdm_jax.flows_affine.spline_flow`::

    Z ~ N(0, 1)  --spline-->  Y  --loc(ctx) + scale(ctx) * Y-->  log T  --exp-->  T

with the log-input scaling applied to ``context`` inside the graph, so the ONNX model takes
the raw parameters in the sidecar's ``context_names`` order. The clamp of the inputs to the
training box that the likelihood applies is *not* part of the flow and is not exported; a
caller outside the box gets the flow's extrapolation.

Run (defaults to the final CRDM conditioner, dt = 0.0005)::

    python scripts/export_conditioner_onnx.py --out spline_flow.onnx

jax2onnx 0.16 fails on jax 0.11 (``Var.__init__() takes 2 positional arguments``); 0.17 works.
Until ``uv.lock`` is refreshed, ``uv run --with jax2onnx==0.17.0 python scripts/...`` runs it
without touching the environment.

or name another checkpoint::

    python scripts/export_conditioner_onnx.py --ckpt outputs/<model>/<overrides>/conditioner
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.scipy import stats

from eamax.flows.checkpoint import read_metadata

# The private helpers are used so the exported bijector is built exactly as the likelihood
# builds it, rather than from a copy that could drift.
from confrdm_jax.flows_affine import (
    _layout_from_width,
    _loc_scale,
    _spline,
    input_scaling,
    load_conditioner,
    make_mlp_conditioner,
    spline_settings,
)

ROOT = Path(__file__).resolve().parents[1]

FINAL_CRDM_CONDITIONER = (
    ROOT
    / "outputs/crdm/model.flow_affine=true/model.flow_log_inputs=true/model.flow_num_hidden=2"
    / "model.num_bins=12/model.num_mid=128/model.sampler.dt=0.0005"
    / "optimizer=adam_cosine_decay_clip/train_steps=100000/conditioner"
)


def load_from_sidecar(path, step=0):
    """Load a conditioner, building its template from the checkpoint's sidecar."""
    meta = read_metadata(path)
    if meta is None:
        raise FileNotFoundError(f"No conditioner sidecar at {path}")
    template = make_mlp_conditioner(
        nnx.Rngs(default=0),
        num_in=meta["num_in"],
        num_mid=meta["num_mid"],
        num_bins=meta["num_bins"],
        affine=meta.get("affine", False),
        spline_range=meta.get("spline_range", 5.0),
        boundary_slopes=meta.get("boundary_slopes", "identity"),
        input_scaling=meta.get("input_scaling"),
        num_hidden=meta.get("num_hidden", 1),
    )
    conditioner = load_conditioner(template, str(path), step=step, context_names=meta["context_names"])
    conditioner.eval()
    return conditioner, list(meta["context_names"])


#: Numerical Recipes' Chebyshev fit, ``erfc(x) = t exp(-x^2 + P(t))`` with ``t = 1 / (1 + x/2)``
#: for ``x >= 0``; fractional error below 1.2e-7 everywhere.
_ERFC_COEFFS = (
    -1.26551223,
    1.00002368,
    0.37409196,
    0.09678418,
    -0.18628806,
    0.27886807,
    -1.13520398,
    1.48851587,
    -0.82215223,
    0.17087277,
)


def _log_erfc_nonneg(x):
    t = 1.0 / (1.0 + 0.5 * x)
    poly = jnp.zeros_like(t)
    for c in reversed(_ERFC_COEFFS):
        poly = poly * t + c
    return jnp.log(t) - x * x + poly


def log_sf_normal(z):
    """Standard normal log-survival, stable in float32 and in the exported graph.

    ``jax.scipy.stats.norm.logsf`` lowers to an ONNX graph that loses precision past
    ``z ~ 5`` and returns ``-inf`` past ``z ~ 5.6`` in float32 -- exactly the tail that scores
    censored trials by ``log S(t_max)``. Here ``log erfc`` is formed in log space, so it
    neither underflows nor cancels. Agrees with ``norm.logsf`` to 1.1e-7 (absolute, float64)
    over ``z`` in [-8, 40].
    """
    log_half_erfc = jnp.log(0.5) + _log_erfc_nonneg(jnp.abs(z) / jnp.sqrt(2.0).astype(z.dtype))
    return jnp.where(z >= 0.0, log_half_erfc, jnp.log1p(-jnp.exp(log_half_erfc)))


def build_jax_fn(conditioner):
    """(data, context) -> (log_pdf, log_sf).

    Conditioner (NNX) runs natively on the batched `context` outside vmap, since the
    jax2onnx-patched `nnx.linear` primitive has no vmap batching rule. The distrax
    spline math runs per-example under vmap, so distrax-internal shapes (e.g. RQS
    `pad_shape`) stay fully static — avoiding the symbolic-dim crash in `jnp.full`.
    """
    num_bins, affine = _layout_from_width(int(conditioner.linear2.out_features))
    spline_range, boundary_slopes = spline_settings(conditioner)
    scaling = input_scaling(conditioner)

    def flow_logprobs(data_scalar, params):
        """Per-example log_pdf and log_sf — all shapes static.

        The inverse is written out stage by stage rather than through ``distrax.Chain``:
        with the affine stage in the chain, jax2onnx (0.17) emits a Reshape that ORT
        rejects ("Invalid position of 0"). It is the same density as
        :func:`confrdm_jax.flows_affine.spline_flow` (agreement ~3e-14 in float64).
        """
        log_t = jnp.log(data_scalar)
        y, log_det = log_t, -log_t  # log |d log t / d t|
        if affine:
            loc, scale = _loc_scale(params)
            y = (log_t - loc) / scale
            log_det = log_det - jnp.log(scale)
        spline = _spline(params[: 3 * num_bins + 1], spline_range, boundary_slopes)
        z, spline_log_det = spline.inverse_and_log_det(y)
        return stats.norm.logpdf(z) + spline_log_det + log_det, log_sf_normal(z)

    flow_batched = jax.vmap(flow_logprobs, in_axes=(0, 0))

    def fn(data, context):
        if scaling is not None:
            eps, loc, scale = (jnp.asarray(scaling[key], dtype=context.dtype) for key in ("eps", "loc", "scale"))
            context = (jnp.log(context + eps) - loc) / scale
        params = conditioner(context)  # (B, 3K+1) or (B, 3K+3) — NNX matmul, natively batched
        return flow_batched(data, params)

    return fn


def parity_inputs(conditioner, num_context, batch=100, seed=0):
    """Decision times and contexts for the JAX-vs-ONNX parity check.

    For a log-input conditioner the contexts are drawn where it was trained -- standardised
    log inputs within +-1.5 -- because far outside the box float32 log-survivals underflow to
    ``-inf`` and no longer compare. A plain conditioner records no scaling, so its contexts
    fall back to Gamma(2) draws.
    """
    rng = np.random.default_rng(seed)
    data = np.linspace(0.005, 4.0, batch, dtype=np.float32)
    scaling = input_scaling(conditioner)
    if scaling is None:
        context = rng.gamma(shape=2.0, size=(batch, num_context))
    else:
        eps, loc, scale = (np.asarray(scaling[key]) for key in ("eps", "loc", "scale"))
        u = rng.uniform(-1.5, 1.5, size=(batch, num_context))
        context = np.maximum(np.exp(loc + scale * u) - eps, 0.0)
    return data, context.astype(np.float32)


def rewrite_bool_where(onnx_path: Path) -> int:
    """Replace `Where(bool, bool, bool)` with `Or(And(c,x), And(Not c, y))`.

    ORT (through 1.26) does not register a Where kernel for bool tensors. The
    logical equivalent uses only And/Or/Not, all of which have bool kernels.
    Returns the number of nodes rewritten.
    """
    import onnx
    from onnx import helper

    m = onnx.load(str(onnx_path))
    inferred = onnx.shape_inference.infer_shapes(m, strict_mode=False, check_type=False)
    vi = {v.name: v for v in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output)}
    inits = {i.name: i for i in inferred.graph.initializer}

    def dtype_of(name):
        if name in inits:
            return inits[name].data_type
        v = vi.get(name)
        return v.type.tensor_type.elem_type if v else 0

    new_nodes = []
    rewrites = 0
    BOOL = onnx.TensorProto.BOOL
    for n in m.graph.node:
        if n.op_type == "Where" and len(n.input) == 3 and all(dtype_of(i) == BOOL for i in n.input):
            c, x, y = n.input
            out = n.output[0]
            base = n.name or out
            not_c = base + "__notc"
            cx = base + "__cx"
            ncy = base + "__ncy"
            new_nodes.extend(
                [
                    helper.make_node("Not", [c], [not_c], name=base + "__not"),
                    helper.make_node("And", [c, x], [cx], name=base + "__and1"),
                    helper.make_node("And", [not_c, y], [ncy], name=base + "__and2"),
                    helper.make_node("Or", [cx, ncy], [out], name=base + "__or"),
                ]
            )
            rewrites += 1
        else:
            new_nodes.append(n)

    if rewrites:
        del m.graph.node[:]
        m.graph.node.extend(new_nodes)
        onnx.save(m, str(onnx_path))
    return rewrites


def onnx_parity(onnx_path: Path, data, context, jax_out, rtol=1e-4, atol=1e-4):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]
    feed = {
        in_names[0]: np.asarray(data, np.float32),
        in_names[1]: np.asarray(context, np.float32),
    }
    onnx_out = sess.run(None, feed)
    jax_lp, jax_lsf = (np.asarray(o) for o in jax_out)
    onnx_lp, onnx_lsf = onnx_out
    np.testing.assert_allclose(onnx_lp, jax_lp, rtol=rtol, atol=atol)
    np.testing.assert_allclose(onnx_lsf, jax_lsf, rtol=rtol, atol=atol)
    print(
        f"ONNX parity OK (max abs diff: log_pdf={np.max(np.abs(onnx_lp - jax_lp)):.2e}, "
        f"log_sf={np.max(np.abs(onnx_lsf - jax_lsf)):.2e})"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=FINAL_CRDM_CONDITIONER)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("spline_flow.onnx"))
    args = ap.parse_args()

    jax.config.update("jax_enable_x64", False)

    cond, context_names = load_from_sidecar(args.ckpt.absolute(), step=args.step)
    num_context = len(context_names)
    print(f"Loaded {args.ckpt} (context: {', '.join(context_names)})")

    fn = build_jax_fn(cond)

    # JAX sanity pass + reference outputs for ORT parity later.
    data, context = parity_inputs(cond, num_context)
    jax_out = fn(data, context)
    assert all(np.all(np.isfinite(np.asarray(o))) for o in jax_out), "JAX fn produced non-finite outputs"
    print(f"JAX fn OK on dummy batch (data {data.shape}, context {context.shape})")

    from jax2onnx import to_onnx

    to_onnx(
        fn,
        # Two inputs: rt (B,) and raw context (B, num_context), in context_names order. 'B' = symbolic batch dim.
        inputs=[("B",), ("B", num_context)],
        return_mode="file",
        output_path=str(args.out),
        # jax2onnx defaults to opset 23. ORT 1.26 hasn't registered Where[bool]
        # kernels that high yet; 20 is the lowest opset that still contains Gelu.
        opset=20,
    )
    print(f"Wrote ONNX -> {args.out}")

    n_rewrites = rewrite_bool_where(args.out)
    if n_rewrites:
        print(f"Rewrote {n_rewrites} bool Where node(s) as And/Or/Not (ORT has no Where[bool] kernel)")

    onnx_parity(args.out, data, context, jax_out)


if __name__ == "__main__":
    main()
