"""Compare neural NLE densities against reference densities on a uniform parameter grid.

For both the Wald distribution and the conflict Wald distribution, evaluates neural NLE
densities against reference densities, measuring accuracy (MAD between PDF and CDF)
and computational speed.

Usage
-----
python scripts/compare_neural_densities.py \\
    wald.conditioner_path=outputs/rdm/.../conditioner \\
    "crdm.conditioners=[{dt: 0.0005, path: outputs/crdm/.../conditioner}]"
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

from confrdm_jax.flows import load_conditioner, make_mlp_conditioner, spline_flow
from confrdm_jax.likelihoods.crdm_volterra import solve_volterra_fpt
from confrdm_jax.likelihoods.rdm import inv_gauss_logpdf, inv_gauss_logsf

logger = logging.getLogger(__name__)

logging.getLogger("absl").setLevel(logging.ERROR)

jax.config.update('jax_enable_x64', True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RT_GRID = jnp.arange(0.005, 4.005, 0.005)  # 800 points


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


def make_wald_ref_batch(rt_grid):
    @jax.jit
    def wald_ref_batch(params):
        v, s, b = params[:, 0], params[:, 1], params[:, 2]
        mu  = b / v
        lam = (b / s) ** 2
        log_pdf = jax.vmap(lambda m, l: inv_gauss_logpdf(rt_grid, m, l))(mu, lam)
        log_sf  = jax.vmap(lambda m, l: inv_gauss_logsf(rt_grid, m, l))(mu, lam)
        return jnp.exp(log_pdf), 1.0 - jnp.exp(log_sf)

    return wald_ref_batch


def make_volterra_batch_fn(rt_grid, dt, num_steps):
    t_volt = jnp.arange(1, num_steps + 1) * dt

    @jax.jit
    def volterra_batch(params):
        tau_safe = jnp.maximum(params[:, 2], 1e-6)

        def _single(vc, amp, tau, s, b):
            return solve_volterra_fpt(vc, amp, tau, s, b, dt, num_steps)

        g_grids, G_grids = jax.vmap(_single)(
            params[:, 0], params[:, 1], tau_safe, params[:, 3], params[:, 4]
        )
        pdf = jax.vmap(lambda g: jnp.interp(rt_grid, t_volt, g))(g_grids)
        cdf = jax.vmap(lambda G: jnp.interp(rt_grid, t_volt, G))(G_grids)
        return jnp.maximum(pdf, 1e-30), jnp.clip(cdf, 0.0, 1.0)

    return volterra_batch


def make_neural_batch_fn(conditioner, rt_grid):
    conditioner.eval()

    @nnx.jit
    def neural_batch(params):
        def single(ctx):
            _, flow = spline_flow(rt_grid, ctx, conditioner)
            z = flow.bijector.inverse(rt_grid)
            return flow.log_prob(rt_grid), stats.norm.logsf(z)

        log_pdfs, log_sfs = jax.lax.map(single, params)
        return jnp.exp(log_pdfs), 1.0 - jnp.exp(log_sfs)

    return neural_batch


def _mean_abs_diff(a, b):
    return jnp.mean(jnp.abs(a - b), axis=1)


def _warmup(fn, params):
    fn(params)[0].block_until_ready()


def _time_fn(fn, params):
    t0 = time.perf_counter()
    result = fn(params)
    result[0].block_until_ready()
    return result, time.perf_counter() - t0


def _load_conditioner(path_str, num_bins, num_mid, num_in):
    path = Path(path_str)
    if not path.exists():
        warnings.warn(f"Conditioner path not found, skipping: {path}")
        return None

    rngs = nnx.Rngs(default=0)
    model = make_mlp_conditioner(rngs, num_in=num_in, num_mid=num_mid, num_bins=num_bins)
    return load_conditioner(model, str(path.absolute()))


def run_wald_comparison(conditioner_path, num_bins, num_mid, wald_cfg):
    conditioner = _load_conditioner(conditioner_path, num_bins, num_mid, num_in=3)
    if conditioner is None:
        return None

    rt_grid = RT_GRID
    params  = make_wald_grid(wald_cfg)

    ref_fn    = make_wald_ref_batch(rt_grid)
    neural_fn = make_neural_batch_fn(conditioner, rt_grid)

    logger.info("Warming up...")
    _warmup(ref_fn, params)
    _warmup(neural_fn, params)

    (ref_pdf, ref_cdf) = ref_fn(params)
    (nn_pdf,  nn_cdf)  = neural_fn(params)

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
        "mad_pdf":              np.array(_mean_abs_diff(nn_pdf, ref_pdf)),
        "mad_cdf":              np.array(_mean_abs_diff(nn_cdf, ref_cdf)),
        "time_ref_per_param":   time_ref_per,
        "time_neural_per_param": time_nn_per,
    }


def run_crdm_comparison(conditioner_path, dt, num_bins, num_mid, chunk_size, crdm_cfg):
    conditioner = _load_conditioner(conditioner_path, num_bins, num_mid, num_in=5)
    if conditioner is None:
        return None

    rt_grid   = RT_GRID
    t_max     = float(rt_grid[-1])
    num_steps = round(t_max / dt)
    params    = make_crdm_grid(crdm_cfg)
    n_total   = params.shape[0]

    ref_fn    = make_volterra_batch_fn(rt_grid, dt, num_steps)
    neural_fn = make_neural_batch_fn(conditioner, rt_grid)

    logger.info("Warming up...")
    warmup_chunk = params[:chunk_size]
    _warmup(ref_fn, warmup_chunk)
    _warmup(neural_fn, warmup_chunk)

    mad_pdf_all  = np.zeros(n_total, dtype=np.float64)
    mad_cdf_all  = np.zeros(n_total, dtype=np.float64)
    time_ref_per = np.zeros(n_total, dtype=np.float64)
    time_nn_per  = np.zeros(n_total, dtype=np.float64)

    for i in range(0, n_total, chunk_size):
        chunk = params[i : i + chunk_size]
        sz    = chunk.shape[0]
        end   = i + sz

        (ref_pdf, ref_cdf), t_ref = _time_fn(ref_fn, chunk)
        (nn_pdf,  nn_cdf),  t_nn  = _time_fn(neural_fn, chunk)

        mad_pdf_all[i:end]  = np.array(_mean_abs_diff(nn_pdf, ref_pdf))
        mad_cdf_all[i:end]  = np.array(_mean_abs_diff(nn_cdf, ref_cdf))

        time_ref_per[i:end] = t_ref / sz
        time_nn_per[i:end]  = t_nn  / sz

    return {
        "params":               np.array(params),
        "mad_pdf":              mad_pdf_all,
        "mad_cdf":              mad_cdf_all,
        "time_ref_per_param":   time_ref_per,
        "time_neural_per_param": time_nn_per,
        "dt":                   dt,
    }

def _result_to_dataset(result, param_names):
    coords = {"param": np.arange(result["params"].shape[0])}

    for j, name in enumerate(param_names):
        coords[name] = ("param", result["params"][:, j])

    data_vars = {
        "mad_pdf":               ("param", result["mad_pdf"]),
        "mad_cdf":               ("param", result["mad_cdf"]),
        "time_ref_per_param":    ("param", result["time_ref_per_param"]),
        "time_neural_per_param": ("param", result["time_neural_per_param"]),
    }

    attrs = {"dt": result["dt"]} if "dt" in result else {}
    return xr.Dataset(data_vars, coords=coords, attrs=attrs)

@hydra.main(version_base=None, config_path="../conf_jax", config_name="compare_densities")
def main(cfg: DictConfig) -> None:
    tree_dict: dict[str, xr.Dataset] = {}

    # --- Wald ---
    if cfg.wald.conditioner_path is not None:
        logger.info(f"\n=== Wald  [{cfg.wald.conditioner_path}] ===")
        result = run_wald_comparison(
            cfg.wald.conditioner_path,
            num_bins=cfg.wald.num_bins,
            num_mid=cfg.wald.num_mid,
            wald_cfg=cfg.wald.param_values,
        )
        if result is not None:
            tree_dict["wald"] = _result_to_dataset(result, ["v", "s", "b"])
    else:
        logger.info("No Wald conditioner specified.")

    # --- CRDM ---
    if cfg.crdm.conditioners:
        for entry in cfg.crdm.conditioners:
            node_name = f"crdm_dt{entry.dt}"
            logger.info(f"\n=== CRDM dt={entry.dt}  [{entry.path}] ===")
            result = run_crdm_comparison(
                entry.path,
                dt=entry.dt,
                num_bins=cfg.crdm.num_bins,
                num_mid=cfg.crdm.num_mid,
                chunk_size=cfg.evaluation.chunk_size,
                crdm_cfg=cfg.crdm.param_values,
            )
            if result is not None:
                tree_dict[node_name] = _result_to_dataset(
                    result, ["v_c", "amp", "tau", "s", "b"]
                )
    else:
        logger.info("No CRDM conditioners specified.")

    xr.DataTree.from_dict(tree_dict).to_netcdf("outputs/compare_densities/density_comparison.nc")
    logger.info("\nResults saved")


if __name__ == "__main__":
    main()
