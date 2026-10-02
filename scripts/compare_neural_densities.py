"""Compare neural NLE densities against reference densities on a uniform parameter grid.

For both the Wald distribution and the conflict Wald distribution, evaluates neural NLE
densities against reference densities (analytic inverse Gaussian and the Volterra solver),
measuring accuracy and computational speed.

**Which Volterra reference.** Accuracy is always measured against the Volterra solver at
``crdm.reference_dt`` (default 0.0005), whatever ``dt`` a flow was trained at, so every
flow is scored against the same, trustworthy density. Matching the flow's own ``dt`` does
not work at coarse steps: at ``dt = 0.05`` the solver loses 20-100% of the mass for fast
parameter sets (most crossings within the first step), which would be scored as flow error.
Timing, by contrast, uses the solver at the flow's own ``dt`` -- the cost of the reference
the flow stands in for.

Accuracy metrics, per parameter set, with the reference as ``P`` and the flow as ``Q``:

* ``kl`` -- KL(P || Q) in nats, i.e. the expected per-trial log-likelihood error. The
  evaluation grid covers ``[T_MIN, T_MAX]``; the mass outside it enters as two censored
  terms, ``F_P(T_MIN) log(F_P / F_Q)`` below and ``S_P(T_MAX) log(S_P / S_Q)`` above, the
  same survival term that scores censored trials in the likelihood. Coarsening the ends
  can only lower the KL, so this is a (tight) lower bound on the full KL. A reference that
  is not exactly normalised can give small negative values.
* ``w1`` -- Wasserstein-1 distance, ``int |F_P - F_Q| dt`` over ``[0, T_MAX]``, in seconds:
  how far the flow moves probability mass in time.
* ``ks`` -- Kolmogorov-Smirnov distance, ``max |F_P - F_Q|`` over the grid.

All three are invariant to the grid, unlike the grid-mean absolute deviations this script
used to report. The grid is log-spaced, so fast, narrow densities get as many points as
slow, wide ones.

Usage
-----
python scripts/compare_neural_densities.py \\
    wald.conditioner_path=outputs/rdm/.../conditioner \\
    "crdm.conditioners=[{dt: 0.0005, path: outputs/crdm/.../conditioner}]"

Each conditioner is rebuilt from its checkpoint's sidecar (depth, affine layout, spline
settings, log-input scaling, width, bins) and evaluated through
:mod:`confrdm_jax.flows_affine`, so plain and affine / log-input / deep flows both work.
"""

import logging
import time
import warnings
from pathlib import Path

import hydra
import xarray as xr
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.scipy import stats
from omegaconf import DictConfig

from eamax.accumulators import inv_gauss_logpdf, inv_gauss_logsf, solve_volterra_fpt
from eamax.flows.checkpoint import read_metadata

from confrdm_jax import configure_jax
from confrdm_jax.flows_affine import load_conditioner, make_mlp_conditioner, spline_flow
from confrdm_jax.specs import CRDM_CONTEXT_NAMES, WALD_CONTEXT_NAMES

logger = logging.getLogger(__name__)

logging.getLogger("absl").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

T_MIN = 1e-3
T_MAX = 4.0
RT_GRID = jnp.geomspace(T_MIN, T_MAX, 2000)

# Floor on densities and probabilities before taking logs.
_TINY = 1e-300


def make_wald_grid(wald_cfg):
    v = wald_cfg["v"]
    s = wald_cfg["s"]
    b = wald_cfg["b"]
    grids = np.meshgrid(v, s, b, indexing="ij")
    return jnp.array(np.stack([g.ravel() for g in grids], axis=1))


def make_crdm_grid(crdm_cfg):
    v_c = crdm_cfg["v"]
    amp = crdm_cfg["amp"]
    tau = crdm_cfg["tau"]
    s = crdm_cfg["s"]
    b = crdm_cfg["b"]
    grids = np.meshgrid(v_c, amp, tau, s, b, indexing="ij")
    return jnp.array(np.stack([g.ravel() for g in grids], axis=1))


# Every batch function returns ``(log_pdf, log_cdf, log_sf)``, each ``(num_params, num_t)``.


def make_wald_ref_batch(rt_grid):
    @jax.jit
    def wald_ref_batch(params):
        v, s, b = params[:, 0], params[:, 1], params[:, 2]
        mu  = b / v
        lam = (b / s) ** 2
        log_pdf = jax.vmap(lambda m, l: inv_gauss_logpdf(rt_grid, m, l))(mu, lam)
        log_sf  = jax.vmap(lambda m, l: inv_gauss_logsf(rt_grid, m, l))(mu, lam)
        log_cdf = jnp.log(jnp.maximum(-jnp.expm1(log_sf), _TINY))
        return log_pdf, log_cdf, log_sf

    return wald_ref_batch


def make_volterra_batch_fn(rt_grid, dt, num_steps):
    # The solver's grid starts at dt; anchor it at (0, 0) so decision times below dt are
    # interpolated towards zero density rather than held at g(dt), which at dt = 0.05 is a
    # visible error on the first grid cell.
    t_volt = jnp.arange(0, num_steps + 1) * dt

    @jax.jit
    def volterra_batch(params):
        tau_safe = jnp.maximum(params[:, 2], 1e-6)

        def _single(vc, amp, tau, s, b):
            return solve_volterra_fpt(vc, amp, tau, s, b, dt, num_steps)

        g_grids, G_grids = jax.vmap(_single)(
            params[:, 0], params[:, 1], tau_safe, params[:, 3], params[:, 4]
        )
        g_grids = jnp.pad(g_grids, ((0, 0), (1, 0)))
        G_grids = jnp.pad(G_grids, ((0, 0), (1, 0)))
        pdf = jax.vmap(lambda g: jnp.interp(rt_grid, t_volt, g))(g_grids)
        cdf = jnp.clip(jax.vmap(lambda G: jnp.interp(rt_grid, t_volt, G))(G_grids), 0.0, 1.0)
        return (
            jnp.log(jnp.maximum(pdf, _TINY)),
            jnp.log(jnp.maximum(cdf, _TINY)),
            jnp.log(jnp.maximum(1.0 - cdf, _TINY)),
        )

    return volterra_batch


def make_neural_batch_fn(conditioner, rt_grid):
    conditioner.eval()

    @nnx.jit
    def neural_batch(params):
        def single(ctx):
            _, flow = spline_flow(rt_grid, ctx, conditioner)
            z = flow.bijector.inverse(rt_grid)
            return flow.log_prob(rt_grid), stats.norm.logcdf(z), stats.norm.logsf(z)

        return jax.lax.map(single, params)

    return neural_batch


@jax.jit
def divergences(ref, neural, t):
    """KL(ref || neural) with censored ends, W1 and KS, one value per parameter set."""
    lp_ref, lF_ref, lS_ref = ref
    lp_nn, lF_nn, lS_nn = neural

    p_ref = jnp.exp(lp_ref)
    body = jnp.trapezoid(jnp.where(p_ref > 0.0, p_ref * (lp_ref - lp_nn), 0.0), t, axis=1)
    below = jnp.exp(lF_ref[:, 0]) * (lF_ref[:, 0] - lF_nn[:, 0])
    above = jnp.exp(lS_ref[:, -1]) * (lS_ref[:, -1] - lS_nn[:, -1])
    kl = below + body + above

    abs_diff = jnp.abs(jnp.exp(lF_ref) - jnp.exp(lF_nn))
    # [0, T_MIN] closed with a triangle: both CDFs start at 0.
    w1 = jnp.trapezoid(abs_diff, t, axis=1) + 0.5 * t[0] * abs_diff[:, 0]
    ks = jnp.max(abs_diff, axis=1)
    return kl, w1, ks


def _warmup(fn, params):
    fn(params)[0].block_until_ready()


def _time_fn(fn, params):
    t0 = time.perf_counter()
    result = fn(params)
    result[0].block_until_ready()
    return result, time.perf_counter() - t0


def _load_conditioner(path_str, context_names):
    """Load a checkpoint, building its template from the sidecar.

    The sidecar records the architecture, so the template always matches the weights. The
    conditioning set is still checked against `context_names`: nothing in the weights records
    what the context columns mean, so a reordered set of the same width would otherwise load
    without complaint and produce silently wrong densities.
    """
    path = Path(path_str).absolute()
    if not path.exists():
        warnings.warn(f"Conditioner path not found, skipping: {path}")
        return None

    meta = read_metadata(path)
    if meta is None:
        raise FileNotFoundError(f"No conditioner sidecar at {path}")
    model = make_mlp_conditioner(
        nnx.Rngs(default=0),
        num_in=len(context_names),
        num_mid=meta["num_mid"],
        num_bins=meta["num_bins"],
        affine=meta.get("affine", False),
        spline_range=meta.get("spline_range", 5.0),
        boundary_slopes=meta.get("boundary_slopes", "identity"),
        input_scaling=meta.get("input_scaling"),
        num_hidden=meta.get("num_hidden", 1),
    )
    return load_conditioner(model, str(path), context_names=list(context_names))


def run_wald_comparison(conditioner_path, wald_cfg):
    conditioner = _load_conditioner(conditioner_path, WALD_CONTEXT_NAMES)
    if conditioner is None:
        return None

    rt_grid = RT_GRID
    params  = make_wald_grid(wald_cfg)

    ref_fn    = make_wald_ref_batch(rt_grid)
    neural_fn = make_neural_batch_fn(conditioner, rt_grid)

    logger.info("Warming up...")
    _warmup(ref_fn, params)
    _warmup(neural_fn, params)

    kl, w1, ks = divergences(ref_fn(params), neural_fn(params), rt_grid)

    # Warmup single-param shape before individual timing loop
    _warmup(ref_fn, params[:1])
    _warmup(neural_fn, params[:1])

    n_total = params.shape[0]
    time_ref_per = np.zeros(n_total, dtype=np.float64)
    time_nn_per  = np.zeros(n_total, dtype=np.float64)
    for i in range(n_total):
        p = params[i : i + 1]
        _, time_ref_per[i] = _time_fn(ref_fn, p)
        _, time_nn_per[i]  = _time_fn(neural_fn, p)

    return {
        "params":               np.array(params),
        "kl":                   np.array(kl),
        "w1":                   np.array(w1),
        "ks":                   np.array(ks),
        "time_ref_per_param":   time_ref_per,
        "time_neural_per_param": time_nn_per,
    }


def compute_volterra_reference(crdm_cfg, reference_dt, chunk_size):
    """The Volterra reference over the whole parameter grid, once, for every flow.

    Returns ``((log_pdf, log_cdf, log_sf), time_per_param)``; the arrays are
    ``(num_params, num_t)`` NumPy arrays in :func:`make_crdm_grid` order.
    """
    params  = make_crdm_grid(crdm_cfg)
    n_total = params.shape[0]
    ref_fn  = make_volterra_batch_fn(RT_GRID, reference_dt, round(T_MAX / reference_dt))

    logger.info("Volterra reference at dt=%s over %d parameter sets", reference_dt, n_total)
    _warmup(ref_fn, params[:chunk_size])

    parts = [[], [], []]
    time_per = np.zeros(n_total, dtype=np.float64)
    for i in range(0, n_total, chunk_size):
        chunk = params[i : i + chunk_size]
        ref, t_ref = _time_fn(ref_fn, chunk)
        for part, value in zip(parts, ref):
            part.append(np.asarray(value))
        time_per[i : i + chunk.shape[0]] = t_ref / chunk.shape[0]
    return tuple(np.concatenate(part) for part in parts), time_per


def run_crdm_comparison(conditioner_path, dt, reference, reference_dt, chunk_size, crdm_cfg):
    """Score one flow against the shared `reference`; time the solver at the flow's `dt`.

    `reference` is :func:`compute_volterra_reference`'s result at `reference_dt`. When `dt`
    equals `reference_dt` its timings are reused instead of solving again.
    """
    conditioner = _load_conditioner(conditioner_path, CRDM_CONTEXT_NAMES)
    if conditioner is None:
        return None

    rt_grid   = RT_GRID
    params    = make_crdm_grid(crdm_cfg)
    n_total   = params.shape[0]
    ref_arrays, ref_times = reference
    time_reference = dt != reference_dt

    neural_fn = make_neural_batch_fn(conditioner, rt_grid)
    timing_ref_fn = make_volterra_batch_fn(rt_grid, dt, round(T_MAX / dt))

    logger.info("Warming up...")
    warmup_chunk = params[:chunk_size]
    _warmup(neural_fn, warmup_chunk)
    if time_reference:
        _warmup(timing_ref_fn, warmup_chunk)

    kl_all       = np.zeros(n_total, dtype=np.float64)
    w1_all       = np.zeros(n_total, dtype=np.float64)
    ks_all       = np.zeros(n_total, dtype=np.float64)
    time_ref_per = ref_times.copy()
    time_nn_per  = np.zeros(n_total, dtype=np.float64)

    for i in range(0, n_total, chunk_size):
        chunk = params[i : i + chunk_size]
        sz    = chunk.shape[0]
        end   = i + sz

        nn, t_nn = _time_fn(neural_fn, chunk)
        if time_reference:
            _, t_ref = _time_fn(timing_ref_fn, chunk)
            time_ref_per[i:end] = t_ref / sz

        ref = tuple(jnp.asarray(a[i:end]) for a in ref_arrays)
        kl, w1, ks = divergences(ref, nn, rt_grid)
        kl_all[i:end] = np.array(kl)
        w1_all[i:end] = np.array(w1)
        ks_all[i:end] = np.array(ks)

        time_nn_per[i:end]  = t_nn  / sz

    return {
        "params":               np.array(params),
        "kl":                   kl_all,
        "w1":                   w1_all,
        "ks":                   ks_all,
        "time_ref_per_param":   time_ref_per,
        "time_neural_per_param": time_nn_per,
        "dt":                   dt,
        "reference_dt":         reference_dt,
    }


def _result_to_dataset(result, param_names):
    coords = {"param": np.arange(result["params"].shape[0])}

    for j, name in enumerate(param_names):
        coords[name] = ("param", result["params"][:, j])

    data_vars = {
        "kl":                    ("param", result["kl"], {"units": "nats"}),
        "w1":                    ("param", result["w1"], {"units": "s"}),
        "ks":                    ("param", result["ks"]),
        "time_ref_per_param":    ("param", result["time_ref_per_param"], {"units": "s"}),
        "time_neural_per_param": ("param", result["time_neural_per_param"], {"units": "s"}),
    }

    attrs = {"t_min": T_MIN, "t_max": T_MAX, "num_t": int(RT_GRID.shape[0])}
    for key in ("dt", "reference_dt"):
        if key in result:
            attrs[key] = result[key]
    return xr.Dataset(data_vars, coords=coords, attrs=attrs)


@hydra.main(version_base=None, config_path="../conf_jax", config_name="compare_densities")
def main(cfg: DictConfig) -> None:
    configure_jax(cfg.get("device", "auto"), require_device=cfg.get("require_device", True))

    tree_dict: dict[str, xr.Dataset] = {}

    # --- Wald ---
    if cfg.wald.conditioner_path is not None:
        logger.info(f"\n=== Wald  [{cfg.wald.conditioner_path}] ===")
        result = run_wald_comparison(
            cfg.wald.conditioner_path,
            wald_cfg=cfg.wald.param_values,
        )
        if result is not None:
            tree_dict["wald"] = _result_to_dataset(result, ["v", "s", "b"])
    else:
        logger.info("No Wald conditioner specified.")

    # --- CRDM ---
    if cfg.crdm.conditioners:
        reference_dt = float(cfg.crdm.reference_dt)
        reference = compute_volterra_reference(
            cfg.crdm.param_values, reference_dt, cfg.evaluation.chunk_size,
        )
        for entry in cfg.crdm.conditioners:
            node_name = f"crdm_dt{entry.dt}"
            logger.info(f"\n=== CRDM dt={entry.dt}  [{entry.path}] ===")
            result = run_crdm_comparison(
                entry.path,
                dt=entry.dt,
                reference=reference,
                reference_dt=reference_dt,
                chunk_size=cfg.evaluation.chunk_size,
                crdm_cfg=cfg.crdm.param_values,
            )
            if result is not None:
                tree_dict[node_name] = _result_to_dataset(
                    result, ["v_c", "amp", "tau", "s", "b"]
                )
    else:
        logger.info("No CRDM conditioners specified.")

    xr.DataTree.from_dict(tree_dict).to_netcdf(cfg.output_path)
    logger.info("\nResults saved to %s", cfg.output_path)


if __name__ == "__main__":
    main()
