"""Simulation-based calibration (SBC) figures for the single-subject RDM and CRDM.

No new fits are needed: ``scripts/parameter_recovery.py`` already does what SBC asks for.
It draws every data set's true parameters from ``model.recovery_prior`` and fits them under
``log_prior_fn`` of that same prior. The PIT of each data set and parameter is therefore
the posterior mass below the generating value, ``mean(draws < true)``. If the posterior is
calibrated, the PITs are uniform on [0, 1].

Each figure is a grid of trials x parameters showing the PIT ECDF minus the uniform CDF
(the Delta-ECDF), with the 95% simultaneous band of Säilynoja et al. (2022) for that many
independent uniforms. The RDM overlays the neural (``approx``) and analytic (``ref``)
posteriors. The analytic curve checks the sampler and this pipeline, so a departure that
appears only in the neural curve is the flow's. The CRDM has no analytic likelihood and
shows the neural curve only.

Unlike the recovery figures, **no data set is masked** for R-hat or ESS: SBC is only valid
over the full set of prior draws, and dropping fits conditions on the data. The number of
data sets failing the recovery figures' convergence thresholds is printed instead. As in
the CRDM recovery figures, the CRDM's ``t0`` is not shown.

Reads the same runs as ``create_figures_parameter_recovery_single.py``, whose loading and
plotting helpers it reuses, and writes ``figures/sbc_single_{rdm,crdm}_{RUN_TAG}.png``. The
trial counts and the convergence thresholds it reports against are the single-subject ones
in ``conf_jax/figures.yaml``.

Run from anywhere:
    python scripts/create_figures_sbc_single.py
"""

import sys
from pathlib import Path
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from arviz_stats.ecdf_utils import ecdf_pit
from matplotlib.lines import Line2D
from xarray import open_datatree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import create_figures_parameter_recovery_single as recovery  # noqa: E402

from confrdm_jax import runs  # noqa: E402

SETTINGS = runs.figure_settings()
NUM_OBS = SETTINGS["num_obs"]
RHAT_THRESHOLD = SETTINGS["convergence"]["single"]["rhat"]
ESS_THRESHOLD = SETTINGS["convergence"]["single"]["ess"]
PALETTE = SETTINGS["palette"]

CI_PROB = 0.95
NUM_BAND_SIMULATIONS = 1000

KIND_LABELS = {"approx": "Neural", "ref": "Analytic"}
KIND_COLORS = {"approx": PALETTE[1], "ref": PALETTE[0]}
# The neural curve is the one under test, so it is drawn over the analytic one.
KIND_ZORDER = {"approx": 3, "ref": 2}


# --- Loading --------------------------------------------------------------------------


def count_unconverged(dt, param_names):
    """Number of data sets failing R-hat or bulk/tail ESS on any parameter."""
    posterior = dt["posterior"].ds[param_names]
    rhat_ok = (az.rhat(posterior).to_array(dim="param") < RHAT_THRESHOLD).all(dim="param")
    ess_ok = xr.concat(
        [
            az.ess(posterior, method=method).to_array(dim="param") > ESS_THRESHOLD
            for method in ["bulk", "tail"]
        ],
        dim="method",
    ).all(dim=["method", "param"])
    return int((~(rhat_ok & ess_ok)).sum())


def load_pit(model, kind):
    """PIT per parameter, as a Dataset of ``(num_obs, subject)`` variables.

    Returns the Dataset and the number of unconverged data sets per ``num_obs``.
    """
    pits, unconverged = [], {}
    for t in NUM_OBS:
        dt = open_datatree(recovery.recovery_nc(model, t, kind))
        param_names = list(dt.attrs["param_names"])
        posterior = dt["posterior"].ds[param_names]
        theta = dt["constant_data"].ds["theta"]
        pits.append(
            xr.Dataset(
                {p: (posterior[p] < theta.sel(param=p, drop=True)).mean(dim=["chain", "draw"]) for p in param_names},
            ),
        )
        unconverged[t] = count_unconverged(dt, param_names)
    ds = xr.concat(pits, dim=xr.DataArray(NUM_OBS, dims="num_obs"))
    return ds, unconverged


def report(model, kind, ds, unconverged):
    print(f"{model} {kind}:")
    for t in NUM_OBS:
        pit = ds.sel(num_obs=t)
        means = ", ".join(f"{p}={pit[p].mean().item():.3f}" for p in ds.data_vars)
        print(
            f"  {t:>4} trials: {pit.sizes['subject']} data sets, "
            f"{unconverged[t]} unconverged; mean PIT {means}",
        )


# --- Plotting -------------------------------------------------------------------------


def pit_ecdf_df(pits, labels, exclude=()):
    """Long frame of Delta-ECDF curves and bands, one row per evaluation point.

    Columns: ``num_obs``, ``param`` (relabelled), ``kind``, ``x``, ``delta`` (ECDF minus
    uniform CDF), and ``lower`` / ``upper`` (the simultaneous band, also minus the CDF).
    """
    frames = []
    for kind, ds in pits.items():
        for p in ds.data_vars:
            if p in exclude:
                continue
            for t in NUM_OBS:
                vals = np.sort(ds[p].sel(num_obs=t).values)
                x, ecdf, lower, upper = ecdf_pit(vals, CI_PROB, NUM_BAND_SIMULATIONS)
                frames.append(
                    pd.DataFrame(
                        {
                            "num_obs": t,
                            "param": labels.get(p, p),
                            "kind": kind,
                            "x": x,
                            "delta": ecdf - x,
                            "lower": lower - x,
                            "upper": upper - x,
                        },
                    ),
                )
    return pd.concat(frames, ignore_index=True)


def draw_delta_ecdf(data, **kwargs):
    """One panel: the band once (it depends only on the number of PITs), then each curve."""
    ax = plt.gca()
    band = data[data["kind"] == data["kind"].iloc[0]]
    ax.fill_between(band["x"], band["lower"], band["upper"], step="post", color="0.85", lw=0)
    for kind, curve in data.groupby("kind"):
        ax.step(curve["x"], curve["delta"], where="post", color=KIND_COLORS[kind], lw=1.5, zorder=KIND_ZORDER[kind])
    ax.axhline(0, color="k", lw=1, ls="--", zorder=1)


def plot_sbc(df, height=3):
    """Delta-ECDF grid of trials x parameters, laid out like the recovery figures."""
    sns.set_theme(context="paper", style="ticks", font_scale=1.5)
    g = sns.FacetGrid(
        df,
        row="num_obs",
        col="param",
        col_order=list(dict.fromkeys(df["param"])),
        sharex=True,
        sharey=True,
        margin_titles=True,
        height=height,
    )
    g.figure.subplots_adjust(wspace=0, hspace=0)
    g.map_dataframe(draw_delta_ecdf)
    g.set(xlim=(0, 1), xticks=[0, 0.5, 1])
    g.set_xlabels(label="PIT")
    g.set_ylabels(label=r"$\Delta$ ECDF")
    g.set_titles(col_template="{col_name}", row_template="{row_name} trials")
    kinds = list(dict.fromkeys(df["kind"]))
    if len(kinds) > 1:
        g.add_legend({KIND_LABELS[k]: Line2D([], [], color=KIND_COLORS[k], lw=1.5) for k in kinds})
    return g


def run(model, kinds, labels, exclude=(), height=3):
    pits = {}
    for kind in kinds:
        pits[kind], unconverged = load_pit(model, kind)
        report(model, kind, pits[kind], unconverged)
    g = plot_sbc(pit_ecdf_df(pits, labels, exclude), height=height)
    recovery.save(g, f"sbc_single_{model}")


if __name__ == "__main__":
    recovery.OUTDIR.mkdir(parents=True, exist_ok=True)
    run("rdm", ["approx", "ref"], recovery.RDM_PARAM_LABELS)
    # t0 is left out, as in the CRDM recovery figures.
    run("crdm", ["approx"], recovery.CRDM_PARAM_LABELS, exclude=("t0",), height=2.2)
