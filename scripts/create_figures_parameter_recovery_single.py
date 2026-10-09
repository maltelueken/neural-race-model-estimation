"""Single-subject parameter-recovery figures for the RDM and CRDM.

Reads the final run's single-subject recoveries (``conf_jax/experiment/final.yaml``), from
``multirun/<model>/<run_tag>/test_num_obs=<N>/``, found through :mod:`confrdm_jax.runs`;
see ``slurm/parameter_recovery_{rdm,crdm}_single.sh``.

RDM: the neural (``approx``) posterior is compared against the true generating
parameters and against the analytic (``ref``) posterior. CRDM: there is no analytic
likelihood, so the neural posterior is compared against the true parameters only.

Subjects whose R-hat is at or above ``RHAT_THRESHOLD``, or whose bulk or tail ESS is at
or below ``ESS_THRESHOLD``, for any parameter are masked out before summarizing. Both come
from ``conf_jax/figures.yaml`` (``convergence.single``), as do the trial counts.

Outputs (written to ``figures/``, each file name ending in
``_{RUN_TAG}.png`` so the figures from earlier runs are not overwritten):
  * parameter_recovery_single_rdm
  * parameter_cross_recovery_single_rdm
  * posterior_contraction_single_rdm
  * coverage_single_rdm
  * c2st_single_rdm                    (skipped until slurm/c2st_recovery_single.sh has run)
  * parameter_recovery_single_crdm
  * posterior_contraction_single_crdm
  * coverage_single_crdm

The neural-vs-analytic posterior-median correlation table is printed as LaTeX.

Run from anywhere:
    python scripts/create_figures_parameter_recovery_single.py
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from xarray import open_datatree

from confrdm_jax import runs
from confrdm_jax.plots import save_figure

SETTINGS = runs.figure_settings()
NUM_OBS = SETTINGS["num_obs"]
RHAT_THRESHOLD = SETTINGS["convergence"]["single"]["rhat"]
ESS_THRESHOLD = SETTINGS["convergence"]["single"]["ess"]
PALETTE = SETTINGS["palette"]

OUTDIR = runs.ROOT / "figures"
RUN_TAG = runs.run_tag()

# Written by slurm/c2st_recovery_single.sh next to the test_num_obs=* runs.
C2ST_PATH = runs.run_dir("rdm", multirun=True) / "c2st.csv"


def recovery_nc(model, num_obs, kind):
    """One single-subject recovery file: ``kind`` is ``approx`` (neural) or ``ref``."""
    return runs.run_dir(model, f"test_num_obs={num_obs}", multirun=True) / f"parameter_recovery_{kind}.nc"

RDM_PARAM_LABELS = {
    "v_intercept": r"$\nu_\text{intercept}$",
    "v_slope": r"$\nu_\text{slope}$",
    "s_true": r"$s_\text{true}$",
    "t0": r"$t_0$",
}

CRDM_PARAM_LABELS = {
    "v_c_intercept": r"$\nu_\text{intercept}$",
    "v_c_slope": r"$\nu_\text{slope}$",
    "amp": r"$\zeta$",
    "tau": r"$\tau$",
    "s_true": r"$s_\text{true}$",
    "t0": r"$t_0$",
}

RDM_LIMS = {
    "v_intercept": (0.25, 2.0),
    "v_slope": (0.25, 3.0),
    "s_true": (0.25, 2.5),
    "b": (0.25, 3.0),
    "t0": (0, 0.8),
}

# Keyed by label, since the CRDM recovery frame is relabelled before plotting.
CRDM_LIMS = {
    r"$\nu_\text{intercept}$": (0.25, 2.0),
    r"$\nu_\text{slope}$": (1.0, 4.0),
    r"$\zeta$": (0.1, 0.5),
    r"$\tau$": (0, 0.3),
    r"$s_\text{true}$": (0.25, 2.0),
    "b": (0.25, 1.5),
    r"$t_0$": (0, 0.8),
}


# --- Loading --------------------------------------------------------------------------


def mask_bad_subjects(dt, threshold=RHAT_THRESHOLD, ess_threshold=ESS_THRESHOLD):
    """Set every subject failing R-hat or bulk/tail ESS on any parameter to NaN."""
    param_names = list(dt.attrs["param_names"])
    posterior = dt["posterior"].ds[param_names]
    rhat_ok = (az.rhat(posterior).to_array(dim="param") < threshold).all(dim="param")
    ess_ok = xr.concat(
        [az.ess(posterior, method=method).to_array(dim="param") > ess_threshold for method in ["bulk", "tail"]],
        dim="method",
    ).all(dim=["method", "param"])
    valid = rhat_ok & ess_ok

    print(
        f"R-hat ok: {rhat_ok.sum().item()}, ESS ok: {ess_ok.sum().item()}, "
        f"converged subjects: {valid.sum().item()} ({valid.mean().item():.1%})"
    )

    new_dt = xr.DataTree.from_dict(
        {
            "posterior": dt["posterior"].ds.where(valid),
            "constant_data": dt["constant_data"].ds.where(valid),
            "prior": dt["prior"].ds,
        }
    )
    new_dt.attrs = dt.attrs
    return new_dt


def merge_datatrees(dt_dict):
    """Concatenate the per-``num_obs`` trees along a new ``num_obs`` dimension."""
    num_obs_coord = xr.DataArray(NUM_OBS, dims="num_obs")
    dt_merged = xr.DataTree.from_dict(
        {
            group: xr.concat([dt_dict[t][group].ds for t in NUM_OBS], dim=num_obs_coord)
            for group in ["posterior", "constant_data", "prior"]
        }
    )
    dt_merged.attrs = dt_dict[NUM_OBS[0]].attrs
    return dt_merged


def load(model, kind):
    dt_dict = {t: mask_bad_subjects(open_datatree(recovery_nc(model, t, kind))) for t in NUM_OBS}
    return merge_datatrees(dt_dict)


# --- Summaries ------------------------------------------------------------------------


def stack_posterior(dt):
    """Posterior draws as ``theta`` next to the generating values as ``true``."""
    param_names = list(dt.attrs["param_names"])
    return (
        dt["posterior"]
        .ds[param_names]
        .to_array(dim="param")
        .to_dataset(name="theta")
        .merge(dt["constant_data"].ds.rename({"theta": "true"}))
    )


def merge_prior(posterior_stacked, dt):
    param_names = list(dt.attrs["param_names"])
    return posterior_stacked.merge(dt["prior"].ds[param_names].to_array(dim="param").to_dataset(name="prior"))


def calc_posterior_median_df(posterior_stacked):
    return posterior_stacked.median(dim=["chain", "draw"]).to_dataframe().reset_index()


def calc_posterior_ci_df(posterior_stacked, q_low=0.025, q_high=0.975):
    quantiles = posterior_stacked.theta.quantile([q_low, 0.5, q_high], dim=["chain", "draw"])
    return (
        xr.Dataset(
            {
                "lower": quantiles.sel(quantile=q_low, drop=True),
                "theta": quantiles.sel(quantile=0.5, drop=True),
                "upper": quantiles.sel(quantile=q_high, drop=True),
            }
        )
        .merge(posterior_stacked.true)
        .to_dataframe()
        .reset_index()
    )


def calc_posterior_median_true_correlation(df):
    return df.drop(["subject"], axis=1).groupby(["num_obs", "param"]).corr().iloc[0::2, -1].reset_index()


def calc_posterior_contraction_df(dt):
    posterior_var = dt.theta.var(dim=["chain", "draw"])
    prior_var = dt.prior.var(dim=["draw"])
    return (
        (1.0 - (posterior_var / prior_var))
        .to_dataset(name="contraction")
        .merge(dt.mean(dim=["chain", "draw"]))
        .to_dataframe()
        .reset_index()
    )


def calc_coverage_df(dt):
    lower = dt.quantile(dim=["chain", "draw"], q=0.025)
    upper = dt.quantile(dim=["chain", "draw"], q=0.975)
    true = dt.true
    covered = ((true > lower) & (true < upper)).theta.astype(float)
    covered = covered.where(lower.theta.notnull())
    return covered.mean(dim="subject", skipna=True).to_dataset(name="coverage").to_dataframe().reset_index()


# --- Plotting -------------------------------------------------------------------------


def scatter_with_ci(data, color=None, **kwargs):
    ax = plt.gca()
    yerr = np.vstack([data["theta"] - data["lower"], data["upper"] - data["theta"]])
    ax.errorbar(
        data["true"],
        data["theta"],
        yerr=yerr,
        fmt="o",
        markersize=3,
        color=color,
        elinewidth=0.5,
        ecolor="gray",
        alpha=0.6,
        zorder=1,
    )


def set_identity_lims(g, lims, df_cor=None):
    """Equal x/y limits and an identity line per panel, optionally annotated with r."""
    hide_inner_xticklabels(g)
    for (row_val, col_val), ax in g.axes_dict.items():
        ax.set_xlim(lims[col_val])
        ax.set_ylim(lims[col_val])
        ax.plot(lims[col_val], lims[col_val], "k--", lw=1, zorder=2)
        if df_cor is not None:
            cor = np.round(
                df_cor[(df_cor["num_obs"] == row_val) & (df_cor["param"] == col_val)]["true"].item(),
                2,
            )
            ax.text(0.1, 0.9, f"r = {cor}", transform=ax.transAxes)


def hide_inner_xticklabels(g):
    for (row_val, _), ax in g.axes_dict.items():
        if row_val != NUM_OBS[-1]:
            ax.set_xticklabels([])


def save(fig_or_grid, name):
    out = OUTDIR / f"{name}_{RUN_TAG}.png"
    save_figure(fig_or_grid, out, dpi=100)
    print(f"saved {out}")


# --- RDM: neural vs. true and vs. analytic --------------------------------------------


def rdm():
    dt_merged_approx = load("rdm", "approx")
    dt_merged_ref = load("rdm", "ref")

    posterior_stacked_approx = stack_posterior(dt_merged_approx)
    posterior_stacked_ref = stack_posterior(dt_merged_ref)

    df_median_approx = calc_posterior_median_df(posterior_stacked_approx)
    df_median_ref = calc_posterior_median_df(posterior_stacked_ref)
    df_median_join = df_median_approx.merge(
        df_median_ref,
        on=["subject", "num_obs", "param"],
        suffixes=["_approx", "_ref"],
    )
    df_ci_approx = calc_posterior_ci_df(posterior_stacked_approx)
    df_cor_approx = calc_posterior_median_true_correlation(df_median_approx)

    # Recovery: neural posterior median vs. true parameter.
    sns.set_theme(context="paper", style="ticks", font_scale=1.5)
    g = sns.FacetGrid(
        df_ci_approx,
        row="num_obs",
        col="param",
        sharex=False,
        sharey=False,
        margin_titles=True,
    )
    g.figure.subplots_adjust(wspace=0, hspace=0)
    g.map_dataframe(scatter_with_ci)
    g.set_xlabels(label="True parameter")
    g.set_ylabels(label="Posterior median")
    g.set_titles(col_template="{col_name}", row_template="{row_name} trials")
    set_identity_lims(g, RDM_LIMS, df_cor_approx)
    save(g, "parameter_recovery_single_rdm")

    # Neural vs. analytic posterior-median correlations.
    df_cor_join = (
        df_median_join.drop(["subject"], axis=1)
        .groupby(["num_obs", "param"])
        .corr()
        .iloc[0::4, -2]
        .reset_index()
        .pivot(index="num_obs", columns="param", values="theta_ref")
        .round(4)
    )
    print(df_cor_join.to_latex(float_format="%.4f"))

    # Cross recovery: neural vs. analytic posterior median.
    sns.set_theme(context="paper", style="ticks", font_scale=1.5)
    g = sns.FacetGrid(
        df_median_join,
        row="num_obs",
        col="param",
        sharex=False,
        sharey=False,
        margin_titles=True,
    )
    g.figure.subplots_adjust(wspace=0, hspace=0)
    g.map_dataframe(sns.scatterplot, x="theta_ref", y="theta_approx")
    g.set_xlabels(label="Posterior median\n(analytic)")
    g.set_ylabels(label="Posterior median\n(neural)")
    g.set_titles(col_template="{col_name}", row_template="{row_name} trials")
    set_identity_lims(g, RDM_LIMS)
    save(g, "parameter_cross_recovery_single_rdm")

    # Posterior contraction.
    df_contraction_approx = calc_posterior_contraction_df(merge_prior(posterior_stacked_approx, dt_merged_approx))
    df_contraction_ref = calc_posterior_contraction_df(merge_prior(posterior_stacked_ref, dt_merged_ref))
    df_contraction_join = df_contraction_approx.merge(
        df_contraction_ref,
        on=["subject", "num_obs", "param", "true"],
        suffixes=["_approx", "_ref"],
    ).melt(
        id_vars=["subject", "num_obs", "param", "true"],
        value_vars=["contraction_approx", "contraction_ref"],
        var_name="method",
    )
    df_contraction_join["param"] = df_contraction_join["param"].replace(RDM_PARAM_LABELS)

    g = sns.FacetGrid(
        df_contraction_join,
        row="num_obs",
        col="param",
        sharex=False,
        sharey=True,
        margin_titles=True,
        height=2.2,
    )
    g.figure.subplots_adjust(wspace=0, hspace=0)
    g.map_dataframe(sns.scatterplot, x="true", y="value", hue="method", alpha=0.5, palette=PALETTE)
    g.set_xlabels(label="True parameter")
    g.set_ylabels(label="Contraction")
    g.set_titles(col_template="{col_name}", row_template="{row_name} trials")
    g.add_legend(
        {
            "Analytic": g._legend_data["contraction_ref"],
            "Neural": g._legend_data["contraction_approx"],
        }
    )
    hide_inner_xticklabels(g)
    save(g, "posterior_contraction_single_rdm")

    # Coverage of the 95% central interval.
    df_coverage_join = (
        calc_coverage_df(posterior_stacked_approx)
        .merge(
            calc_coverage_df(posterior_stacked_ref),
            on=["num_obs", "param"],
            suffixes=["_approx", "_ref"],
        )
        .melt(
            id_vars=["num_obs", "param"],
            value_vars=["coverage_approx", "coverage_ref"],
            var_name="method",
        )
    )
    df_coverage_join["param"] = df_coverage_join["param"].replace(RDM_PARAM_LABELS)

    sns.set_theme(context="paper", style="ticks", font_scale=2.0)
    g = sns.FacetGrid(df_coverage_join, col="param", sharex=False, sharey=True, margin_titles=True)
    g.figure.subplots_adjust(wspace=0, hspace=0)
    g.map_dataframe(sns.barplot, x="num_obs", y="value", hue="method", palette=PALETTE)
    g.refline(y=0.95)
    g.set(yticks=[0, 0.5, 1.0])
    g.set_xlabels(label="Trials")
    g.set_ylabels(label="Coverage")
    g.set_titles(col_template="{col_name}")
    g.add_legend(
        {
            "Neural": g._legend_data["coverage_approx"],
            "Analytic": g._legend_data["coverage_ref"],
        }
    )
    save(g, "coverage_single_rdm")

    # C2ST accuracy between neural and analytic posteriors.
    if not C2ST_PATH.exists():
        print(f"skipping C2ST figure: {C2ST_PATH} not found")
        return
    df_c2st = pd.read_csv(C2ST_PATH)
    sns.set_theme(style="ticks", rc={"axes.spines.right": False, "axes.spines.top": False})
    fig, ax = plt.subplots()
    sns.boxplot(df_c2st, x="num_obs", y="c2st_accuracy", ax=ax, palette=[PALETTE[1]])
    ax.hlines(0.5, xmin=-0.5, xmax=3.5, linestyles="--", colors="grey")
    ax.set_xlim((-0.5, 3.5))
    ax.set_xlabel("Number of trials")
    ax.set_ylabel("C2ST accuracy")
    save(fig, "c2st_single_rdm")


# --- CRDM: neural vs. true ------------------------------------------------------------


def crdm():
    dt_merged_approx = load("crdm", "approx")
    posterior_stacked_approx = stack_posterior(dt_merged_approx)

    df_median_approx = calc_posterior_median_df(posterior_stacked_approx)
    df_ci_approx = calc_posterior_ci_df(posterior_stacked_approx)
    df_cor_approx = calc_posterior_median_true_correlation(df_median_approx)

    df_ci_approx["param"] = df_ci_approx["param"].replace(CRDM_PARAM_LABELS)
    df_cor_approx["param"] = df_cor_approx["param"].replace(CRDM_PARAM_LABELS)
    t0_label = CRDM_PARAM_LABELS["t0"]

    # Recovery: neural posterior median vs. true parameter (t0 omitted).
    sns.set_theme(context="paper", style="ticks", font_scale=1.5)
    g = sns.FacetGrid(
        df_ci_approx.loc[df_ci_approx["param"] != t0_label, :],
        row="num_obs",
        col="param",
        sharex=False,
        sharey=False,
        margin_titles=True,
        height=2.2,
    )
    g.figure.subplots_adjust(wspace=0, hspace=0)
    g.map_dataframe(scatter_with_ci, color=PALETTE[1])
    g.set_xlabels(label="True parameter")
    g.set_ylabels(label="Posterior median")
    g.set_titles(col_template="{col_name}", row_template="{row_name} trials")
    set_identity_lims(g, CRDM_LIMS, df_cor_approx)
    save(g, "parameter_recovery_single_crdm")

    # Posterior contraction (t0 omitted).
    df_contraction_approx = calc_posterior_contraction_df(merge_prior(posterior_stacked_approx, dt_merged_approx))
    df_contraction_approx["contraction"] = np.clip(df_contraction_approx["contraction"], 0, 1)
    df_contraction_approx["param"] = df_contraction_approx["param"].replace(CRDM_PARAM_LABELS)

    g = sns.FacetGrid(
        df_contraction_approx.loc[df_contraction_approx["param"] != t0_label, :],
        row="num_obs",
        col="param",
        sharex=False,
        sharey=True,
        margin_titles=True,
        height=2.2,
    )
    g.figure.subplots_adjust(wspace=0, hspace=0)
    g.map_dataframe(
        sns.scatterplot,
        x="true",
        y="contraction",
        hue="num_obs",
        alpha=0.5,
        palette=[PALETTE[0]],
    )
    g.set_xlabels(label="True parameter")
    g.set_ylabels(label="Contraction")
    g.set_titles(col_template="{col_name}", row_template="{row_name} trials")
    hide_inner_xticklabels(g)
    save(g, "posterior_contraction_single_crdm")

    # Coverage of the 95% central interval.
    df_coverage_approx = calc_coverage_df(posterior_stacked_approx)
    df_coverage_approx["param"] = df_coverage_approx["param"].replace(CRDM_PARAM_LABELS)

    sns.set_theme(context="paper", style="ticks", font_scale=2.0)
    g = sns.FacetGrid(df_coverage_approx, col="param", sharex=False, sharey=True, margin_titles=True)
    g.figure.subplots_adjust(wspace=0, hspace=0)
    g.map_dataframe(sns.barplot, x="num_obs", y="coverage", hue="num_obs", palette=[PALETTE[0]])
    g.refline(y=0.95)
    g.set(yticks=[0, 0.5, 1.0])
    g.set_xlabels(label="Trials")
    g.set_ylabels(label="Coverage")
    g.set_titles(col_template="{col_name}")
    save(g, "coverage_single_crdm")


if __name__ == "__main__":
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rdm()
    crdm()
