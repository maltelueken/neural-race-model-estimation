"""Compare posterior parameter correlations between neural and analytic
posteriors for the single-subject RDM parameter-recovery results.

The other single-subject figures compare *recovery* correlations (posterior
median vs. true parameter). This script instead looks at the *posterior
correlation structure*: for each simulated subject we estimate the pairwise
Pearson correlations among the RDM parameters from the posterior samples, and
compare the neural (spline-flow / NLE, ``approx``) posterior against the
analytic (inverse-Gaussian likelihood, ``ref``) posterior.

Reads the recoveries of the final RDM flow, trained for 100k steps with an affine layer, two
hidden layers, log-scaled inputs and gradient clipping on the raised-minimum box (s, b in
[0.25, 3.5]); see ``slurm/parameter_recovery_rdm_multirun_affine_log_deep_clip_box.sh``.

Outputs (written to ``figures/``, each file name ending in ``_{RUN_TAG}.png``):
  * posterior_correlation_scatter_single_rdm
        analytic vs. neural posterior correlation, one panel per parameter pair
  * posterior_correlation_error_single_rdm
        distribution of (neural - analytic) correlation error vs. trials

Run from anywhere:
    python scripts/create_figure_posterior_correlation_single.py
"""

import itertools
from pathlib import Path

import arviz as az
import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from xarray import open_datatree

jax.config.update("jax_enable_x64", True)

NUM_OBS = [50, 250, 500, 1000]
RHAT_THRESHOLD = 1.05

ROOT = Path(__file__).resolve().parents[1]
MULTIRUN = ROOT / "multirun"
OUTDIR = ROOT / "figures"

RUN_TAG = "affine_log_deep_clip_100k"

# The recoveries were launched with model.flow_affine=True / model.flow_log_inputs=True,
# so Hydra spelled the directory names with "True".
RUN_DIR = (
    MULTIRUN
    / "rdm/model.flow_affine=True/model.flow_log_inputs=True/model.flow_num_hidden=2"
    / "model.num_bins=12/model.num_mid=128"
    / "model.training_prior.b_max=3.5/model.training_prior.b_min=0.25"
    / "model.training_prior.s_max=3.5/model.training_prior.s_min=0.25"
    / "optimizer=adam_cosine_decay_clip"
)
TRAIN_STEPS = 100000

# Neural (spline-flow NLE) is "approx", analytic (inverse-Gaussian) is "ref".
METHOD_LABELS = {"approx": "Neural", "ref": "Analytic"}

PARAM_LABELS = {
    "v_intercept": r"$\nu_\mathrm{intercept}$",
    "v_slope": r"$\nu_\mathrm{slope}$",
    "s_true": r"$s_\mathrm{true}$",
    "b": r"$b$",
    "t0": r"$t_0$",
}


def nc_path(kind, num_obs):
    return (
        RUN_DIR
        / f"test_num_obs={num_obs}/train_steps={TRAIN_STEPS}"
        / f"parameter_recovery_{kind}.nc"
    )


def valid_subjects(dt, threshold=RHAT_THRESHOLD):
    """Boolean DataArray over ``subject``: True where all params converged."""
    param_names = list(dt.param_names)
    rhat_ds = az.rhat(dt["posterior"].ds[param_names])
    return (rhat_ds.to_array(dim="param") < threshold).all(dim="param")


def posterior_correlations(ds, param_names, keep_subjects):
    """Pairwise posterior Pearson correlations per subject.

    Returns a long-form DataFrame with columns ``subject``, ``pair``, ``corr``.
    Subjects not in ``keep_subjects`` (or containing NaNs) are skipped.
    """
    arr = (
        ds[param_names]
        .to_array(dim="param")
        .stack(sample=("chain", "draw"))
        .transpose("subject", "sample", "param")
    )
    data = arr.values  # (n_subject, n_sample, n_param)
    subjects = arr["subject"].values

    pairs = list(itertools.combinations(range(len(param_names)), 2))
    keep = set(np.asarray(keep_subjects).tolist())

    records = []
    for si, subj in enumerate(subjects):
        if subj not in keep:
            continue
        x = data[si]  # (n_sample, n_param)
        if not np.isfinite(x).all():
            continue
        corr = np.corrcoef(x, rowvar=False)  # (n_param, n_param)
        for i, j in pairs:
            pair_label = f"{PARAM_LABELS[param_names[i]]}, {PARAM_LABELS[param_names[j]]}"
            records.append(
                {"subject": int(subj), "pair": pair_label, "corr": corr[i, j]}
            )
    return pd.DataFrame(records)


def build_correlation_df():
    """Join neural and analytic posterior correlations across all trial counts.

    For each trial count, only subjects that converged under *both* methods are
    used, so the neural/analytic comparison is on a common subject set.
    """
    frames = []
    for t in NUM_OBS:
        dt_approx = open_datatree(nc_path("approx", t))
        dt_ref = open_datatree(nc_path("ref", t))

        param_names = list(dt_approx.param_names)

        valid = (valid_subjects(dt_approx) & valid_subjects(dt_ref)).compute()
        common = valid["subject"].values[valid.values]
        print(f"num_obs={t:>4}: {len(common)} subjects converged under both methods")

        df_approx = posterior_correlations(
            dt_approx["posterior"].ds, param_names, common
        )
        df_ref = posterior_correlations(dt_ref["posterior"].ds, param_names, common)

        df = df_approx.merge(
            df_ref, on=["subject", "pair"], suffixes=["_approx", "_ref"]
        )
        df["num_obs"] = t
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df["error"] = df["corr_approx"] - df["corr_ref"]
    return df


def plot_correlation_scatter(df):
    """Analytic vs. neural posterior correlation, one panel per parameter pair."""
    sns.set_theme(context="paper", style="ticks", font_scale=1.5)

    g = sns.FacetGrid(
        df,
        col="pair",
        col_wrap=5,
        hue="num_obs",
        palette="viridis",
        sharex=True,
        sharey=True,
        height=2.4,
    )
    g.map_dataframe(sns.scatterplot, x="corr_ref", y="corr_approx", alpha=0.6, s=18)

    for pair, ax in g.axes_dict.items():
        ax.plot([-1, 1], [-1, 1], "k--", lw=1, zorder=0)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal")
        r = df.loc[df["pair"] == pair, ["corr_ref", "corr_approx"]].corr().iloc[0, 1]
        ax.text(0.05, 0.88, f"r = {r:.2f}", transform=ax.transAxes, fontsize=13)

    g.set_titles(col_template="{col_name}")
    g.set_xlabels("Analytic corr.")
    g.set_ylabels("Neural corr.")
    g.add_legend(title="Trials")

    out = OUTDIR / f"posterior_correlation_scatter_single_rdm_{RUN_TAG}.png"
    g.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(g.figure)


def plot_correlation_error(df):
    """Distribution of (neural - analytic) correlation error by trial count."""
    sns.set_theme(context="paper", style="ticks", font_scale=1.5)

    g = sns.FacetGrid(
        df,
        col="pair",
        col_wrap=5,
        sharey=True,
        height=2.4,
    )
    g.map_dataframe(
        sns.boxplot,
        x="num_obs",
        y="error",
        hue="num_obs",
        palette="viridis",
        legend=False,
        fliersize=1.5,
    )
    g.refline(y=0.0)

    g.set_titles(col_template="{col_name}")
    g.set_xlabels("Trials")
    g.set_ylabels("Neural $-$ analytic")

    out = OUTDIR / f"posterior_correlation_error_single_rdm_{RUN_TAG}.png"
    g.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(g.figure)


def print_error_summary(df):
    """Per-trial-count summary of the correlation discrepancy."""
    summary = (
        df.groupby("num_obs")["error"]
        .agg(
            mean_bias="mean",
            mean_abs_error=lambda s: s.abs().mean(),
            rmse=lambda s: np.sqrt((s**2).mean()),
        )
        .round(4)
    )
    print("\nNeural vs. analytic posterior-correlation discrepancy:")
    print(summary.to_string())


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = build_correlation_df()
    plot_correlation_scatter(df)
    plot_correlation_error(df)
    print_error_summary(df)


if __name__ == "__main__":
    main()
