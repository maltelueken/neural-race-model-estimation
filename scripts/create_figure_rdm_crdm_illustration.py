"""Illustration of evidence accumulation in the RDM and the CRDM.

Two panels share the y-axis:

  * left, the racing diffusion model: two accumulators with constant drifts v_0 and v_1,
  * right, the conflict RDM: the same race with a gamma-shaped conflict pulse eta(t) added
    to accumulator 1's drift (the congruent case), with the pulse itself in an inset.

Each panel shows a handful of simulated trials, each drawn up to its decision, and the mean
(noise-free) trajectory of each accumulator.

The trajectories are drawn with a plain Euler-Maruyama scheme in NumPy, because the figure
needs whole paths and `eamax.simulate` only returns first-passage times. They are an
illustration only: unlike the simulator that trains the flow, there is no Brownian-bridge
crossing correction and the pulse drift is a left Riemann sum.

Output: ``figures/rdm_crdm_illustration.png``

Run from anywhere:
    python scripts/create_figure_rdm_crdm_illustration.py
"""

from pathlib import Path

from confrdm_jax import configure_jax

configure_jax("auto")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
import numpy as np  # noqa: E402
from eamax.accumulators import normalized_gamma_derivative  # noqa: E402
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # noqa: E402

from confrdm_jax.plots import save_figure  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "figures"

# Shared / RDM parameters
V0 = 1.0  # drift rate, accumulator 0 (incorrect / slow response)
V1 = 2.0  # drift rate, accumulator 1 (correct / fast response)
S0 = 1.0  # diffusion coefficient, accumulator 0 (fixed by convention)
S1 = 0.8  # diffusion coefficient, accumulator 1 (the free parameter s_true)
B = 1.5  # decision boundary

# CRDM-only parameters (congruent condition: the pulse aids accumulator 1)
AMP = 0.20  # conflict pulse amplitude
TAU = 0.15  # conflict pulse time constant (s)

# Simulation settings
DT = 0.002
T_MAX = 2.0
N_TRIALS = 10
SEED = 7

C0 = "#D45F5F"  # accumulator 0 (slow / incorrect)
C1 = "#3A7FC1"  # accumulator 1 (fast / correct)
C0_DARK = "#8B1A1A"
C1_DARK = "#1B4F8A"
ALPHA_IND = 0.1  # individual trial transparency
LW_IND = 1.0
LW_MEAN = 2.5


def simulate_trajectories(mu, s, b, n_trials, dt, seed):
    """Euler-Maruyama paths of a two-accumulator race, each up to its decision.

    Parameters
    ----------
    mu : array, shape (2, n_steps)
        Drift of each accumulator at each step.
    s : array, shape (2,)
        Diffusion coefficients.

    Returns
    -------
    list of dict
        Per trial: the full paths ``x`` (2, n_steps), the index ``end_idx`` one past the
        decision, the ``winner`` and the decision time ``rt_decision`` (``inf`` if neither
        accumulator crossed).
    """
    rng = np.random.default_rng(seed)
    n_steps = mu.shape[1]
    trials = []
    for _ in range(n_trials):
        noise = rng.standard_normal((2, n_steps))
        x = np.cumsum(mu * dt + s[:, None] * noise * np.sqrt(dt), axis=1)

        crossed = x >= b
        fpt = np.where(crossed.any(axis=1), (np.argmax(crossed, axis=1) + 1) * dt, np.inf)
        winner = int(np.argmin(fpt))
        rt_decision = float(fpt[winner])
        end_idx = min(int(rt_decision / dt) + 1, n_steps) if rt_decision < np.inf else n_steps

        trials.append({"x": x, "end_idx": end_idx, "winner": winner, "rt_decision": rt_decision})
    return trials


def simulate_rdm(v0, v1, s0, s1, b, n_trials, dt, t_max, seed):
    """RDM trials and the mean trajectories, which are straight lines."""
    t = np.arange(1, int(t_max / dt) + 1) * dt
    mu = np.stack([np.full_like(t, v0), np.full_like(t, v1)])
    trials = simulate_trajectories(mu, np.array([s0, s1]), b, n_trials, dt, seed)
    return trials, t, v0 * t, v1 * t


def simulate_crdm(v0, v1, amp, tau, s0, s1, b, n_trials, dt, t_max, seed):
    """CRDM trials with the conflict pulse on accumulator 1, and the mean trajectories."""
    t = np.arange(1, int(t_max / dt) + 1) * dt
    pulse = np.asarray(normalized_gamma_derivative(t, amp, tau, a_shape=2.0))
    mu = np.stack([np.full_like(t, v0), v1 + pulse])
    trials = simulate_trajectories(mu, np.array([s0, s1]), b, n_trials, dt, seed)
    return trials, t, np.cumsum(mu[0] * dt), np.cumsum(mu[1] * dt)


def draw_accumulation_panel(ax, trials, t, x0_mean, x1_mean, b):
    """Draw the boundary, the individual trials and the mean trajectories of one model."""
    ax.axhline(b, color="#222", lw=1.5, ls="--", zorder=4)

    for tr in trials:
        e, w = tr["end_idx"], tr["winner"]
        loser = 1 - w
        # Losing accumulator behind, winning one on top.
        ax.plot(t[:e], tr["x"][loser, :e], color=(C0 if loser == 0 else C1),
                lw=LW_IND, alpha=ALPHA_IND, zorder=2)
        ax.plot(t[:e], tr["x"][w, :e], color=(C0 if w == 0 else C1),
                lw=LW_IND, alpha=ALPHA_IND + 0.10, zorder=3)

    for x_mean, c_dark in [(x0_mean, C0_DARK), (x1_mean, C1_DARK)]:
        mask = x_mean <= b * 1.005
        ax.plot(t[mask], x_mean[mask], color=c_dark, lw=LW_MEAN, zorder=7)

    ax.text(T_MAX * 0.99, b + 0.04, r"$b$", ha="right", va="bottom", fontweight="bold", color="#222")
    ax.set_xlabel("Decision time (in s)")
    ax.set_xlim(0, T_MAX)


def draw_pulse_inset(ax, t):
    """The conflict pulse eta(t), with its amplitude and time constant marked."""
    ax_inset = inset_axes(
        ax, width="33%", height="38%",
        bbox_to_anchor=(0.65, 0.1, 1, 1), bbox_transform=ax.transAxes, loc="lower left",
    )

    t_plot = t[t <= 0.70]
    pulse = np.asarray(normalized_gamma_derivative(t_plot, AMP, TAU, a_shape=2.0))
    ax_inset.plot(t_plot, pulse, color=C1_DARK, lw=2.0)
    ax_inset.axhline(0, color="gray", lw=0.8, ls="--")

    peak = int(np.argmax(pulse[1:]) + 1)  # skip t = 0
    ax_inset.annotate(
        r"$\zeta=%.2f$" % AMP,
        xy=(t_plot[peak], pulse[peak]),
        xytext=(t_plot[peak] + 0.10, pulse[peak] * 0.65),
        fontsize=10, color=C1_DARK,
        arrowprops=dict(arrowstyle="->", color=C1_DARK, lw=1.0),
    )

    # tau is the pulse's zero crossing.
    ax_inset.axvline(TAU, color="#888", lw=0.9, ls=":")
    ax_inset.annotate(
        r"$\tau = %.2f$ s" % TAU,
        xy=(TAU, 0.0),
        xytext=(TAU + 0.08, 0.7),
        fontsize=10, color=C1_DARK,
        arrowprops=dict(arrowstyle="->", color=C1_DARK, lw=1.0),
    )

    ax_inset.set_title(r"Drift rate $\eta(t)$", fontsize=12)
    ax_inset.set_xlabel("$t$ (s)", fontsize=12)
    ax_inset.set_xticklabels([])
    ax_inset.spines[["top", "right"]].set_visible(False)


def main():
    rdm_trials, t, rdm_x0, rdm_x1 = simulate_rdm(V0, V1, S0, S1, B, N_TRIALS, DT, T_MAX, SEED)
    crdm_trials, _, crdm_x0, crdm_x1 = simulate_crdm(
        V0, V1, AMP, TAU, S0, S1, B, N_TRIALS, DT, T_MAX, SEED
    )
    for name, trials in [("RDM ", rdm_trials), ("CRDM", crdm_trials)]:
        wins = [sum(tr["winner"] == k for tr in trials) for k in (0, 1)]
        print(f"{name} winner counts  acc0:{wins[0]:2d}  acc1:{wins[1]:2d}")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    fig.subplots_adjust(wspace=0.08, left=0.08, right=0.97, top=0.90, bottom=0.12)

    # Left: RDM, with the two constant drifts annotated on the mean trajectories.
    ax = axes[0]
    ax.set_title("Racing diffusion model (RDM)", fontweight="bold", pad=8)
    draw_accumulation_panel(ax, rdm_trials, t, rdm_x0, rdm_x1, B)
    t_ann = 0.35
    ax.annotate(
        r"$v_0 = %.1f$" % V0,
        xy=(t_ann, V0 * t_ann), xytext=(t_ann + 0.25, V0 * t_ann - 0.18),
        fontsize=14, color=C0_DARK, arrowprops=dict(arrowstyle="->", color=C0_DARK, lw=1.4),
    )
    ax.annotate(
        r"$v_1 = %.1f$" % V1,
        xy=(t_ann, V1 * t_ann), xytext=(t_ann + 0.15, V1 * t_ann + 0.12),
        fontsize=14, color=C1_DARK, arrowprops=dict(arrowstyle="->", color=C1_DARK, lw=1.4),
    )
    ax.set_ylabel("Evidence $X(t)$")
    ax.set_ylim(-0.15, B * 1.18)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

    # Right: CRDM, with accumulator 1's pulsed drift annotated at its early steep rise.
    ax = axes[1]
    ax.set_title("Conflict racing diffusion model (CRDM)", fontweight="bold", pad=8)
    draw_accumulation_panel(ax, crdm_trials, t, crdm_x0, crdm_x1, B)
    t_early = 0.18
    ax.annotate(
        r"$v_1 + \eta(t)$",
        xy=(t_early, crdm_x1[int(t_early / DT)]),
        xytext=(t_early + 0.28, crdm_x1[int(t_early / DT)] + 0.05),
        color=C1_DARK,
        arrowprops=dict(arrowstyle="->", color=C1_DARK, lw=1.4, zorder=10),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C1_DARK, alpha=0.85),
        zorder=10,
    )
    t_ann = 0.50
    ax.annotate(
        r"$v_0 = %.1f$" % V0,
        xy=(t_ann, V0 * t_ann), xytext=(t_ann + 0.20, V0 * t_ann - 0.18),
        color=C0_DARK, arrowprops=dict(arrowstyle="->", color=C0_DARK, lw=1.4),
    )
    draw_pulse_inset(ax, t)

    OUTDIR.mkdir(exist_ok=True)
    out = OUTDIR / "rdm_crdm_illustration.png"
    save_figure(fig, out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
