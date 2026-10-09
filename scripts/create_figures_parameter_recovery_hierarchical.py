"""Hierarchical parameter-recovery figures for the RDM and CRDM.

Reads the final run's hierarchical recoveries (``conf_jax/experiment/final.yaml``), which
sit next to its conditioners in ``outputs/<model>/<run_tag>/``, found through
:mod:`confrdm_jax.runs`; see ``slurm/parameter_recovery_{rdm,crdm}_hierarchical.sh``.

RDM: the neural (``approx``) posterior is compared against the analytic (``ref``)
posterior, since both exist. CRDM: there is no analytic likelihood, so the neural
posterior is compared against the true generating parameters.

Parameters whose R-hat (across the SMC chains) is at or above ``RHAT_THRESHOLD``, or
whose bulk or tail ESS is at or below ``ESS_THRESHOLD``, are masked out before
summarizing. Both come from ``conf_jax/figures.yaml`` (``convergence.hierarchical``); the
number of populations from ``conf_jax/config.yaml``.

Outputs (written to ``figures/``, each file name ending in
``_{RUN_TAG}.png`` so the figures from earlier runs are not overwritten):
  * parameter_cross_recovery_hierarchical_subject_rdm
  * parameter_cross_recovery_hierarchical_pop_rdm
  * parameter_recovery_hierarchical_rdm         (subject + population, neural vs. analytic)
  * parameter_recovery_hierarchical_crdm        (subject + population, neural vs. true)
  * parameter_recovery_hierarchical_pop_crdm

Run from anywhere:
    python scripts/create_figures_parameter_recovery_hierarchical.py
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from omegaconf import OmegaConf
from xarray import open_datatree

from confrdm_jax import runs
from confrdm_jax.plots import save_figure

NUM_POPS = OmegaConf.load(runs.CONF_DIR / "config.yaml")["hierarchical_recovery"]["test_num_populations"]
_CONVERGENCE = runs.figure_settings()["convergence"]["hierarchical"]
RHAT_THRESHOLD = _CONVERGENCE["rhat"]
ESS_THRESHOLD = _CONVERGENCE["ess"]
RUN_TAG = runs.run_tag()

OUTDIR = runs.ROOT / "figures"

RUN_DIRS = {model: runs.run_dir(model) for model in ("rdm", "crdm")}

POP_PARAM_LABELS = {
    "theta": r"$\theta$",
    "mu": r"$\mu$",
    "sigma": "s",
}

RDM_PARAM_LABELS = {
    "v_intercept": r"$\nu_\text{intercept}$",
    "v_slope": r"$\nu_\text{slope}$",
    "s_true": r"$s_\text{true}$",
    "b": "b",
    "t0": r"$t_0$",
}

CRDM_PARAM_LABELS = {
    "v_c_intercept": r"$\nu_\text{intercept}$",
    "v_c_slope": r"$\nu_\text{slope}$",
    "amp": r"$\zeta$",
    "tau": r"$\tau$",
    "s_true": r"$s_\text{true}$",
    "b": "b",
    "t0": r"$t_0$",
}

# Axis limits for the subject (theta) and population-mean (mu) rows; the sigma row
# always uses SIGMA_LIMS.
RDM_LIMS = {
    "v_intercept": (0.5, 2.0),
    "v_slope": (1.0, 2.5),
    "s_true": (0.5, 2.0),
    "b": (0.5, 2.0),
    "t0": (0.1, 0.5),
}

CRDM_LIMS = {
    "v_c_intercept": (0.5, 2.3),
    "v_c_slope": (0.5, 4.0),
    "amp": (0.1, 0.5),
    "tau": (0, 0.25),
    "s_true": (0.25, 2.0),
    "b": (0.5, 1.5),
    "t0": (0.15, 0.5),
}

CRDM_POP_LIMS = {
    "v_c_intercept": (0.25, 2.0),
    "v_c_slope": (0.25, 3.0),
    "amp": (0, 0.5),
    "tau": (0, 0.3),
    "s_true": (0.25, 2.0),
    "b": (0.25, 2.0),
    "t0": (0, 0.8),
}

SIGMA_LIMS = (0, 0.5)


# --- Loading --------------------------------------------------------------------------


def load_merged(model, kind):
    """Open the ``NUM_POPS`` recovery files and concatenate them along ``pop``."""
    run_dir = RUN_DIRS[model]
    dts = [open_datatree(run_dir / f"hierarchical_recovery_pop{i}_{kind}.nc") for i in range(NUM_POPS)]
    pop_coord = xr.DataArray(np.arange(NUM_POPS), dims="pop")
    merged = xr.DataTree.from_dict(
        {
            group: xr.concat([dt[group].ds for dt in dts], dim=pop_coord)
            for group in ["posterior", "constant_data", "prior"]
        }
    )
    merged.attrs = dts[0].attrs
    return merged


def mask_bad_params(dt, label, threshold=RHAT_THRESHOLD, ess_threshold=ESS_THRESHOLD):
    """Set every posterior variable failing R-hat or bulk/tail ESS to NaN."""
    posterior = dt["posterior"].ds
    rhat_ds = az.rhat(posterior)
    ess_bulk_ds = az.ess(posterior, method="bulk")
    ess_tail_ds = az.ess(posterior, method="tail")

    print(f"\n{label}: parameters with R-hat < {threshold} and bulk/tail ESS > {ess_threshold}")
    masked = posterior.copy()
    for var in rhat_ds.data_vars:
        rhat_ok = rhat_ds[var] < threshold
        ess_ok = (ess_bulk_ds[var] > ess_threshold) & (ess_tail_ds[var] > ess_threshold)
        valid = rhat_ok & ess_ok
        print(
            f"  {var:>14}: {valid.sum().item():>4} ({valid.mean().item():.1%})"
            f"  [R-hat ok {rhat_ok.sum().item()}, ESS ok {ess_ok.sum().item()}]"
        )
        masked[var] = posterior[var].where(valid)

    new_dt = xr.DataTree.from_dict(
        {
            "posterior": masked,
            "constant_data": dt["constant_data"].ds,
            "prior": dt["prior"].ds,
        }
    )
    new_dt.attrs = dt.attrs
    return new_dt


def load(model, kind):
    return mask_bad_params(load_merged(model, kind), f"{model} {kind}")


# --- Summaries ------------------------------------------------------------------------


def stack_posterior_subject(dt):
    """Subject-level draws as ``theta`` next to the generating values as ``true``."""
    param_names = list(dt.attrs["param_names"])
    return (
        dt["posterior"]
        .ds[param_names]
        .to_array(dim="param")
        .to_dataset(name="theta")
        .merge(dt["constant_data"].ds[["theta"]].rename({"theta": "true"}))
    )


def stack_posterior_pop(dt):
    """Population ``mu`` / ``sigma`` draws next to their generating values.

    ``posterior["mu"]`` is on the natural scale while ``constant_data["mu"]`` is in
    log space (a documented quirk of the stored layout), so the true ``mu`` is
    exponentiated here.
    """
    ds = (
        dt["posterior"]
        .ds[["mu", "sigma"]]
        .to_array(dim="pop_param")
        .to_dataset(name="theta")
        .merge(dt["constant_data"].ds[["mu", "sigma"]].to_array(dim="pop_param").to_dataset(name="true"))
    )
    ds["true"] = xr.where(ds["pop_param"] == "mu", np.exp(ds["true"]), ds["true"])
    return ds


def posterior_median_df(stacked):
    return stacked.median(dim=["chain", "draw"]).to_dataframe().reset_index()


def posterior_ci_df(stacked, q_low=0.025, q_high=0.975):
    quantiles = stacked["theta"].quantile([q_low, 0.5, q_high], dim=["chain", "draw"])
    return (
        xr.Dataset(
            {
                "lower": quantiles.sel(quantile=q_low, drop=True),
                "theta": quantiles.sel(quantile=0.5, drop=True),
                "upper": quantiles.sel(quantile=q_high, drop=True),
            }
        )
        .merge(stacked["true"])
        .to_dataframe()
        .reset_index()
    )


def correlation_table(df, x, y, by):
    """Pearson correlation between columns ``x`` and ``y`` within each ``by`` group."""
    return df.dropna(subset=[x, y]).groupby(by)[[x, y]].apply(lambda g: g[x].corr(g[y])).rename("r")


# --- Plotting -------------------------------------------------------------------------


def add_recovery_ci(data, x="true", **kwargs):
    ax = plt.gca()
    yerr = np.vstack([data["theta"] - data["lower"], data["upper"] - data["theta"]])
    ax.errorbar(
        data[x],
        data["theta"],
        yerr=yerr,
        fmt="none",
        ecolor="gray",
        elinewidth=1.0,
        alpha=0.5,
        zorder=0,
    )


def add_cross_recovery_ci(data, **kwargs):
    ax = plt.gca()
    xerr = np.vstack([data["theta_ref"] - data["lower_ref"], data["upper_ref"] - data["theta_ref"]])
    yerr = np.vstack(
        [
            data["theta_approx"] - data["lower_approx"],
            data["upper_approx"] - data["theta_approx"],
        ]
    )
    ax.errorbar(
        data["theta_ref"],
        data["theta_approx"],
        xerr=xerr,
        yerr=yerr,
        fmt="none",
        ecolor="gray",
        elinewidth=0.5,
        alpha=0.4,
        zorder=0,
    )


def set_identity_lims(g, lims, row_is_sigma=lambda row: False):
    """Set equal x/y limits per panel and draw the identity line."""
    for key, ax in g.axes_dict.items():
        row, col = key if isinstance(key, tuple) else (None, key)
        lim = SIGMA_LIMS if row_is_sigma(row) else lims[col]
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.plot(lim, lim, "k--", lw=1)


def save(g, name):
    out = OUTDIR / f"{name}_{RUN_TAG}.png"
    save_figure(g, out)
    print(f"saved {out}")


def relabel(df, param_labels):
    """Replace parameter names with their LaTeX labels, keeping facet order."""
    df = df.copy()
    df["param"] = pd.Categorical(df["param"].map(param_labels), categories=list(param_labels.values()))
    df["pop_param"] = pd.Categorical(
        df["pop_param"].map(POP_PARAM_LABELS),
        categories=list(POP_PARAM_LABELS.values()),
    )
    return df.dropna(subset=["param"])


# --- RDM: neural vs. analytic ---------------------------------------------------------


def rdm():
    dt_approx = load("rdm", "approx")
    dt_ref = load("rdm", "ref")

    subject_approx = stack_posterior_subject(dt_approx)
    subject_ref = stack_posterior_subject(dt_ref)
    pop_approx = stack_posterior_pop(dt_approx)
    pop_ref = stack_posterior_pop(dt_ref)

    df_ci_subject = posterior_ci_df(subject_approx).merge(
        posterior_ci_df(subject_ref),
        on=["subject", "pop", "param", "true"],
        suffixes=["_approx", "_ref"],
    )
    df_ci_pop = posterior_ci_df(pop_approx).merge(
        posterior_ci_df(pop_ref),
        on=["pop", "param", "pop_param", "true"],
        suffixes=["_approx", "_ref"],
    )

    # Subject-level cross recovery.
    sns.set_theme(context="paper", style="ticks", font_scale=1.2)
    g = sns.FacetGrid(df_ci_subject, col="param", sharex=False, sharey=False, margin_titles=True)
    g.figure.subplots_adjust(wspace=0, hspace=0)
    g.map_dataframe(add_cross_recovery_ci)
    g.map_dataframe(sns.scatterplot, x="theta_ref", y="theta_approx")
    g.set_xlabels(label="Posterior median\n(analytic)")
    g.set_ylabels(label="Posterior median\n(neural)")
    g.set_titles(col_template="{col_name}")
    set_identity_lims(g, RDM_LIMS)
    save(g, "parameter_cross_recovery_hierarchical_subject_rdm")

    # Population-level cross recovery.
    g = sns.FacetGrid(
        df_ci_pop,
        row="pop_param",
        col="param",
        sharex=False,
        sharey=False,
        margin_titles=True,
    )
    g.figure.subplots_adjust(wspace=0, hspace=0)
    g.map_dataframe(add_cross_recovery_ci)
    g.map_dataframe(sns.scatterplot, x="theta_ref", y="theta_approx", s=100.0)
    g.set_xlabels(label="Posterior median\n(analytic)")
    g.set_ylabels(label="Posterior median\n(neural)")
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    set_identity_lims(g, RDM_LIMS, row_is_sigma=lambda row: row == "sigma")
    save(g, "parameter_cross_recovery_hierarchical_pop_rdm")

    # Combined subject + population figure.
    sns.set_theme(context="paper", style="ticks", font_scale=1.5)
    df_ci = relabel(
        pd.concat([df_ci_subject.assign(pop_param="theta"), df_ci_pop]),
        RDM_PARAM_LABELS,
    )
    g = sns.FacetGrid(
        df_ci,
        row="pop_param",
        col="param",
        sharex=False,
        sharey=False,
        margin_titles=True,
        height=2.2,
    )
    g.figure.subplots_adjust(wspace=0, hspace=0)
    g.map_dataframe(add_cross_recovery_ci)
    g.map_dataframe(sns.scatterplot, x="theta_ref", y="theta_approx", hue="pop", palette="Dark2", s=30)
    g.set_xlabels(label="Posterior median\n(analytic)")
    g.set_ylabels(label="Posterior median\n(neural)")
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    set_identity_lims(
        g,
        {RDM_PARAM_LABELS[k]: v for k, v in RDM_LIMS.items()},
        row_is_sigma=lambda row: row == POP_PARAM_LABELS["sigma"],
    )
    save(g, "parameter_recovery_hierarchical_rdm")

    # Neural vs. analytic posterior-median correlations.
    df_median_subject = posterior_median_df(subject_approx).merge(
        posterior_median_df(subject_ref),
        on=["subject", "pop", "param", "true"],
        suffixes=["_approx", "_ref"],
    )
    df_median_pop = posterior_median_df(pop_approx).merge(
        posterior_median_df(pop_ref),
        on=["pop", "param", "pop_param", "true"],
        suffixes=["_approx", "_ref"],
    )
    table = pd.concat(
        [
            correlation_table(
                df_median_subject.assign(pop_param="subject"),
                "theta_approx",
                "theta_ref",
                ["pop_param", "param"],
            ),
            correlation_table(df_median_pop, "theta_approx", "theta_ref", ["pop_param", "param"]),
        ]
    ).unstack("param")[list(RDM_PARAM_LABELS)]
    print("\nRDM: correlation of neural and analytic posterior medians")
    print(table.to_latex(float_format="%.4f"))


# --- CRDM: neural vs. true ------------------------------------------------------------


def crdm():
    dt_approx = load("crdm", "approx")

    subject = stack_posterior_subject(dt_approx)
    pop = stack_posterior_pop(dt_approx)

    df_ci_subject = posterior_ci_df(subject)
    df_ci_pop = posterior_ci_df(pop)

    # Combined subject + population figure (t0 left out, as in the paper figure).
    sns.set_theme(context="paper", style="ticks", font_scale=1.5)
    labels = {k: v for k, v in CRDM_PARAM_LABELS.items() if k != "t0"}
    df_ci = relabel(pd.concat([df_ci_subject.assign(pop_param="theta"), df_ci_pop]), labels)
    g = sns.FacetGrid(
        df_ci,
        row="pop_param",
        col="param",
        sharex=False,
        sharey=False,
        margin_titles=True,
        height=2.2,
    )
    g.figure.subplots_adjust(wspace=0, hspace=0)
    g.map_dataframe(add_recovery_ci, x="true")
    g.map_dataframe(sns.scatterplot, x="true", y="theta", hue="pop", palette="Dark2", s=30)
    g.set_xlabels(label="True parameter")
    g.set_ylabels(label="Posterior median")
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    set_identity_lims(
        g,
        {CRDM_PARAM_LABELS[k]: v for k, v in CRDM_LIMS.items()},
        row_is_sigma=lambda row: row == POP_PARAM_LABELS["sigma"],
    )
    save(g, "parameter_recovery_hierarchical_crdm")

    # Population level only, all parameters.
    g = sns.FacetGrid(
        df_ci_pop,
        row="pop_param",
        col="param",
        sharex=False,
        sharey=False,
        margin_titles=True,
    )
    g.figure.subplots_adjust(wspace=0, hspace=0)
    g.map_dataframe(add_recovery_ci, x="true")
    g.map_dataframe(sns.scatterplot, x="true", y="theta", s=100.0, hue="pop", palette="Dark2")
    g.set_xlabels(label="True parameter")
    g.set_ylabels(label="Posterior median\n(neural)")
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    set_identity_lims(g, CRDM_POP_LIMS, row_is_sigma=lambda row: row == "sigma")
    save(g, "parameter_recovery_hierarchical_pop_crdm")

    # Posterior-median vs. true correlations.
    table = pd.concat(
        [
            correlation_table(
                posterior_median_df(subject).assign(pop_param="subject"),
                "theta",
                "true",
                ["pop_param", "param"],
            ),
            correlation_table(posterior_median_df(pop), "theta", "true", ["pop_param", "param"]),
        ]
    ).unstack("param")[list(CRDM_PARAM_LABELS)]
    print("\nCRDM: correlation of neural posterior medians and true parameters")
    print(table.to_latex(float_format="%.4f"))


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rdm()
    crdm()


if __name__ == "__main__":
    main()
