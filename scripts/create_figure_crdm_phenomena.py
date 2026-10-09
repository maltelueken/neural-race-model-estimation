"""The conflict-task phenomena the CRDM produces, at three settings of the conflict pulse.

For each parameter set one subject's trials are simulated -- half congruent, half
incongruent, sharing one parameter vector -- and three summaries are drawn, one row per set:

  * delta plot: incongruent minus congruent RT quantile (correct trials) against their mean,
  * conditional accuracy function: accuracy per RT quantile bin, per condition,
  * empirical CDF of RT, per condition.

Incongruent trials are red and congruent ones blue (``condition`` is 0 and 1; see
:func:`confrdm_jax.simulators.crdm_condition_design`). The rows vary the pulse time constant
tau, with v_c_slope and b adjusted alongside, to show the delta plot's slope changing sign.

Output: ``figures/crdm_phenomena.png``

Run from anywhere:
    python scripts/create_figure_crdm_phenomena.py
"""

from pathlib import Path

from confrdm_jax import configure_jax

configure_jax("auto")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from confrdm_jax.plots import save_figure  # noqa: E402
from confrdm_jax.simulators import simulate_crdm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "figures"

NUM_TRIALS = 5000
DT = 0.001
T_MAX = 4.0
SEED = 2026

#: Shared by every row.
BASE_PARAMS = {"v_c_intercept": 0.5, "amp": 0.3, "s_true": 0.9, "t0": 0.3}

#: One row each; the remaining parameters come from :data:`BASE_PARAMS`.
ROW_PARAMS = [
    {"v_c_slope": 4.0, "tau": 0.03, "b": 0.7},
    {"v_c_slope": 4.5, "tau": 0.09, "b": 0.9},
    {"v_c_slope": 4.5, "tau": 0.15, "b": 0.7},
]

#: Condition code -> colour: 0 incongruent (red), 1 congruent (blue).
CONDITION_COLORS = {0: "#8B1A1A", 1: "#1B4F8A"}


def simulate_conditions(key, num_trials, *, v_c_intercept, v_c_slope, amp, tau, s_true, b, t0):
    """One subject's trials, half congruent and half incongruent, as ``(rt, resp, cond)``.

    `simulate_crdm` routes the conflict pulse by the trial design's `distractor` column,
    which under this design is the congruency indicator, and returns the condition as the
    third data column.
    """
    log_theta = jnp.log(jnp.array([v_c_intercept, v_c_slope, amp, tau, s_true, b, t0]))
    return np.asarray(simulate_crdm(key, log_theta, num_trials, DT, T_MAX))


def plot_delta(data, ax):
    """Incongruent minus congruent RT quantiles of correct trials, against their mean."""
    rt, resp, cond = data[:, 0], data[:, 1], data[:, 2]
    qs = np.linspace(0, 1, 9)[1:-1]
    quantiles = np.array(
        [
            stats.mstats.mquantiles(rt[(resp == 1) & (cond == i)], qs, alphap=0.5, betap=0.5)
            for i in (0, 1)
        ]
    )
    ax.plot(quantiles.mean(axis=0), quantiles[0] - quantiles[1], "--o", color="black")


def plot_caf(data, ax, n_bins=5):
    """Conditional accuracy function: accuracy per RT quantile bin, per condition."""
    rt, resp, cond = data[:, 0], data[:, 1], data[:, 2]
    for i in np.unique(cond):
        mask = cond == i
        rt_cond, resp_cond = rt[mask], resp[mask]
        edges = np.quantile(rt_cond, np.linspace(0, 1, n_bins + 1))
        bin_index = np.clip(np.digitize(rt_cond, edges, right=True) - 1, 0, n_bins - 1)
        accuracy = [
            resp_cond[bin_index == k].mean() if np.any(bin_index == k) else np.nan
            for k in range(n_bins)
        ]
        ax.plot(range(1, n_bins + 1), accuracy, marker="o", linestyle="-", color=CONDITION_COLORS[int(i)])


def plot_ecdf(data, ax):
    """Empirical CDF of RT, per condition."""
    rt, cond = data[:, 0], data[:, 2]
    for i in np.unique(cond):
        ax.ecdf(rt[cond == i], linestyle="-", color=CONDITION_COLORS[int(i)])


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    key = jax.random.key(SEED)
    fig, axarr = plt.subplots(len(ROW_PARAMS), 3, figsize=(10, 8))
    last = len(ROW_PARAMS) - 1

    for i, (row, axrow) in enumerate(zip(ROW_PARAMS, axarr)):
        # Every row uses the same key, so the rows differ only by their parameters.
        data = simulate_conditions(key, NUM_TRIALS, **BASE_PARAMS, **row)

        plot_delta(data, axrow[0])
        plot_caf(data, axrow[1], n_bins=4)
        plot_ecdf(data, axrow[2])

        axrow[1].set_title(rf"$\tau$ = {row['tau']}")

        axrow[0].set_xlim((0.3, 0.6))
        axrow[0].set_ylim((0, 0.07))
        axrow[0].set_ylabel(r"$\Delta$ (s)")
        axrow[1].set_ylim((0, 1))
        axrow[1].set_ylabel("Accuracy")
        axrow[2].set_ylabel("ECDF")
        if i == last:
            axrow[0].set_xlabel("Mean response time (in s)")
            axrow[1].set_xlabel("Response time bin")
            axrow[2].set_xlabel("Response time (in s)")

    fig.tight_layout()
    OUTDIR.mkdir(exist_ok=True)
    out = OUTDIR / "crdm_phenomena.png"
    save_figure(fig, out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
