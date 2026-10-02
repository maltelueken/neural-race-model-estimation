"""Neural vs. reference density accuracy and timing, from ``compare_neural_densities.py``.

Reads ``outputs/compare_densities/density_comparison.nc``, written for the final run's flows,
and plots, per parameter value:

  * KL(reference || neural) in nats -- the expected per-trial log-likelihood error, with
    the mass beyond the evaluation grid entering as a censored survival term;
  * Wasserstein-1 in milliseconds -- how far the flow moves probability mass in time;
  * Kolmogorov-Smirnov distance -- the largest CDF difference.

Wald (analytic inverse Gaussian reference) and conflict Wald share the figure, coloured by
the ``dt`` each conditioner was trained at. Every conflict-Wald flow is scored against the
same reference, the Volterra solver at ``reference_dt`` (0.0005), so the ``dt`` colour shows
what training on a coarser simulator costs. The timing comparison uses the solver at each
flow's own ``dt`` instead: the cost of the reference that flow replaces.

Outputs (``figures/``):
  * divergence_rdm_crdm_{RUN_TAG}.png
  * timing_comparison_{RUN_TAG}.png

Run from anywhere:
    python scripts/create_figures_density_comparison.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

from confrdm_jax import runs

RUN_TAG = runs.run_tag()

RESULTS = runs.ROOT / "outputs/compare_densities/density_comparison.nc"
OUTDIR = runs.ROOT / "figures"

DT_VALUES = [0.05, 0.005, 0.0005]

METRICS = {
    "kl": "KL (in nats)",
    "w1": r"$W_1$ (in ms)",
    "ks": "KS",
}
PARAM_LABELS = {
    "v": r"$\nu$",
    "v_c": r"$\nu$",
    "amp": r"$\zeta$",
    "tau": r"$\tau$",
    "s": r"$\sigma$",
    "b": r"$b$",
}
WALD_PARAMS = ["v", "s", "b"]
CRDM_PARAMS = ["v_c", "amp", "tau", "s", "b"]


def load():
    results = xr.load_datatree(RESULTS)
    wald = results["wald"].ds
    crdm = {dt: results[f"crdm_dt{dt}"].ds for dt in DT_VALUES}
    return wald, crdm


def long_form(ds, param_names, dt_label):
    """One row per (parameter set, metric, parameter): the value and the parameter value."""
    df = pd.DataFrame({name: ds[name].values for name in param_names})
    for metric in METRICS:
        df[metric] = ds[metric].values
    df["w1"] = df["w1"] * 1e3
    df = df.melt(id_vars=param_names, value_vars=list(METRICS), var_name="metric")
    df = df.melt(
        id_vars=["metric", "value"],
        value_vars=param_names,
        var_name="param",
        value_name="param_value",
    )
    df["param"] = df["param"].map(PARAM_LABELS)
    df["metric"] = df["metric"].map(METRICS)
    df["dt"] = dt_label
    return df


def summarize(wald, crdm):
    print("\nMedian / 90th percentile / max per conditioner:")
    rows = [("Wald", wald)] + [(f"CRDM dt={dt}", crdm[dt]) for dt in DT_VALUES]
    for name, ds in rows:
        parts = []
        for metric, scale, unit in [("kl", 1, ""), ("w1", 1e3, " ms"), ("ks", 1, "")]:
            q = np.percentile(ds[metric].values * scale, [50, 90, 100])
            parts.append(f"{metric} {q[0]:.2g}/{q[1]:.2g}/{q[2]:.2g}{unit}")
        print(f"  {name:>16} (n={ds.sizes['param']:3}): " + "  ".join(parts))


def plot_divergences(wald, crdm):
    frames = [long_form(wald, WALD_PARAMS, "N/A")]
    frames += [long_form(crdm[dt], CRDM_PARAMS, str(dt)) for dt in DT_VALUES]
    df = pd.concat(frames, ignore_index=True)

    sns.set_theme(context="paper", style="ticks", font_scale=1.4)
    g = sns.FacetGrid(
        df,
        row="metric",
        col="param",
        row_order=list(METRICS.values()),
        col_order=[PARAM_LABELS[p] for p in CRDM_PARAMS],
        sharex=False,
        sharey="row",
        margin_titles=True,
        height=2.4,
    )
    g.map_dataframe(
        sns.boxplot,
        x="param_value",
        y="value",
        hue="dt",
        hue_order=["N/A", *map(str, DT_VALUES)],
        palette="Dark2",
        width=0.8,
        fliersize=1.5,
        log_scale=(False, True),
    )
    g.set_xlabels("Parameter value")
    g.set_ylabels("")
    for metric_label, ax in zip(METRICS.values(), g.axes[:, 0]):
        ax.set_ylabel(metric_label)
    g.set_titles(row_template="", col_template="{col_name}")
    g.add_legend(title=r"Training $dt$", loc="lower center", ncols=4, bbox_to_anchor=(0.45, 1.0))

    out = OUTDIR / f"divergence_rdm_crdm_{RUN_TAG}.png"
    g.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(g.figure)


def timing_table(wald, crdm):
    unit = 1e-3  # ms
    rows = []
    for method, var in [("Neural", "time_neural_per_param"), ("Reference", "time_ref_per_param")]:
        rows.append(("Wald", "", method, wald[var].median().item() / unit, wald[var].std().item() / unit))
        for dt in DT_VALUES:
            ds = crdm[dt]
            rows.append(("Conflict", dt, method, ds[var].median().item() / unit, ds[var].std().item() / unit))
    return pd.DataFrame(rows, columns=["Model", "dt", "Method", "Median", "SD"])


def print_timing(df):
    wide = df.pivot(index=["Model", "dt"], columns="Method").swaplevel(axis=1).sort_index(axis=1)
    wide[("", "Ratio")] = wide[("Reference", "Median")] / wide[("Neural", "Median")]
    print("\nMedian evaluation time per parameter set (ms):")
    print(wide.sort_index(ascending=False).to_latex(float_format="%.3f"))


def plot_timing(df):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 16,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 16,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )
    colors = sns.color_palette("Dark2", 3)
    palette = {"Neural": colors[0], "Analytic": colors[1], "Numerical": colors[2]}
    methods = ["Neural", "Reference"]

    def grouped_bar(ax, data, x_order, xticklabels, title, xlabel, method_labels):
        width = 0.8 / len(methods)
        x = np.arange(len(x_order))
        for i, method in enumerate(methods):
            sub = data[data["Method"] == method].set_index("dt").loc[x_order]
            label = method_labels[method]
            ax.bar(
                x + (i - (len(methods) - 1) / 2) * width,
                sub["Median"],
                width,
                yerr=sub["SD"],
                capsize=3,
                label=label,
                color=palette[label],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_yscale("log")
        ax.legend()

    fig, axes = plt.subplots(
        1, 2, figsize=(10, 4.5), sharey=True, gridspec_kw={"width_ratios": [1, 3]}
    )
    grouped_bar(
        axes[0], df[df["Model"] == "Wald"], [""], [""], "Wald", "",
        {"Neural": "Neural", "Reference": "Analytic"},
    )
    grouped_bar(
        axes[1], df[df["Model"] == "Conflict"], DT_VALUES, [str(dt) for dt in DT_VALUES],
        "Conflict", r"$dt$", {"Neural": "Neural", "Reference": "Numerical"},
    )
    axes[0].set_ylabel("Median evaluation\ntime (ms)")
    fig.tight_layout()

    out = OUTDIR / f"timing_comparison_{RUN_TAG}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    wald, crdm = load()
    summarize(wald, crdm)
    plot_divergences(wald, crdm)
    timing = timing_table(wald, crdm)
    print_timing(timing)
    plot_timing(timing)


if __name__ == "__main__":
    main()
