"""Export trained CRDM spline-flow (conditioner + bijector) to ONNX via jax2onnx.

Pipeline: Orbax checkpoint -> JAX function (data, context) -> (log_pdf, log_sf)
       -> jax2onnx.to_onnx -> .onnx

Run:
    pip install jax2onnx onnxruntime
    python scripts/export_conditioner_onnx.py \
        --ckpt outputs/crdm/.../conditioner \
        --num-context 5 --num-bins 12 --num-mid 128 \
        --out spline_flow.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.scipy import stats

import distrax

from confrdm_jax.flows import (
    load_conditioner,
    make_mlp_conditioner,
)


def _spline_logprobs(data_scalar, spline_params_vec):
    """Per-example spline-flow log_pdf and log_sf — all shapes static."""
    spline_layer = distrax.RationalQuadraticSpline(
        spline_params_vec,
        range_min=-5.0,
        range_max=5.0,
        boundary_slopes="identity",
        min_bin_size=1e-4,
    )
    exp_layer = distrax.Lambda(
        forward=jnp.exp,
        inverse=jnp.log,
        forward_log_det_jacobian=lambda z: z,
        inverse_log_det_jacobian=lambda x: -jnp.log(x),
        event_ndims_in=0,
        event_ndims_out=0,
    )
    bijector = distrax.Chain([exp_layer, spline_layer])
    base_dist = distrax.Normal(loc=0.0, scale=1.0)
    flow = distrax.Transformed(base_dist, bijector)
    z = flow.bijector.inverse(data_scalar)
    return flow.log_prob(data_scalar), stats.norm.logsf(z)


def build_jax_fn(conditioner):
    """(data, context) -> (log_pdf, log_sf).

    Conditioner (NNX) runs natively on the batched `context` outside vmap, since the
    jax2onnx-patched `nnx.linear` primitive has no vmap batching rule. The distrax
    spline math runs per-example under vmap, so distrax-internal shapes (e.g. RQS
    `pad_shape`) stay fully static — avoiding the symbolic-dim crash in `jnp.full`.
    """

    _spline_batched = jax.vmap(_spline_logprobs, in_axes=(0, 0))

    def fn(data, context):
        spline_params = conditioner(context)  # (B, 3K+1) — NNX matmul, natively batched
        return _spline_batched(data, spline_params)

    return fn


def parity_inputs(num_context, batch=100, seed=0):
    rng = np.random.default_rng(seed)
    data = jnp.linspace(0.005, 4.0, batch).astype(np.float32)
    context = rng.gamma(shape=2.0, size=(batch, num_context)).astype(np.float32)
    return data, context


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
    vi = {v.name: v for v in list(inferred.graph.value_info)
                          + list(inferred.graph.input)
                          + list(inferred.graph.output)}
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
        if (n.op_type == "Where"
            and len(n.input) == 3
            and all(dtype_of(i) == BOOL for i in n.input)):
            c, x, y = n.input
            out = n.output[0]
            base = n.name or out
            not_c = base + "__notc"
            cx = base + "__cx"
            ncy = base + "__ncy"
            new_nodes.extend([
                helper.make_node("Not", [c], [not_c], name=base + "__not"),
                helper.make_node("And", [c, x], [cx], name=base + "__and1"),
                helper.make_node("And", [not_c, y], [ncy], name=base + "__and2"),
                helper.make_node("Or", [cx, ncy], [out], name=base + "__or"),
            ])
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
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--num-context", type=int, default=5)
    ap.add_argument("--num-bins", type=int, default=12)
    ap.add_argument("--num-mid", type=int, default=128)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("spline_flow.onnx"))
    args = ap.parse_args()

    jax.config.update("jax_enable_x64", False)

    rngs = nnx.Rngs(0)
    cond = make_mlp_conditioner(
        rngs, num_in=args.num_context, num_mid=args.num_mid, num_bins=args.num_bins,
    )
    cond = load_conditioner(cond, str(args.ckpt), step=args.step)
    cond.eval()

    fn = build_jax_fn(cond)

    # JAX sanity pass + reference outputs for ORT parity later.
    data, context = parity_inputs(args.num_context)
    jax_out = fn(data, context)
    assert all(np.all(np.isfinite(np.asarray(o))) for o in jax_out), \
        "JAX fn produced non-finite outputs"
    print(f"JAX fn OK on dummy batch (data {data.shape}, context {context.shape})")

    from jax2onnx import to_onnx

    to_onnx(
        fn,
        # Two inputs: rt (B,) and context (B, num_context). 'B' = symbolic batch dim.
        inputs=[("B",), ("B", args.num_context)],
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
