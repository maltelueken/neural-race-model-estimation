"""Rational-quadratic spline flow: the bijection and the density it produces.

Script version of ``notebooks/create_figure_spline_flow.ipynb``, pointed at the flows trained for 100k
steps with an affine layer, two hidden layers, log-scaled inputs and gradient clipping
(``optimizer=adam_cosine_decay_clip``) -- the same runs as the other figure scripts.

Four panels:

  A. Wald: the flow's bijection, base ``z`` to log first-passage time, with the spline's
     knots, for three drift rates.
  B. Wald: the base density and the flow's first-passage-time density, with the same knots
     mapped to time, against the analytic inverse Gaussian (dotted).
  C, D. The same for the conflict Wald (the CRDM's pulsed accumulator) at three values of
     tau, against the Volterra first-passage-time solver (dotted).

Both flows are affine, so ``log t = loc(ctx) + scale(ctx) * y`` and the knots move with each
parameter set's decision-time distribution rather than sitting at fixed log times.

The conditioner templates are rebuilt from the checkpoints' sidecars (depth, affine layout,
spline settings, log-input scaling), so a checkpoint cannot be loaded into the wrong
architecture.

Output: ``figures/spline_flow_transformation_{RUN_TAG}.png``

Run from anywhere:
    python scripts/create_figure_spline_flow.py
"""

from pathlib import Path

from confrdm_jax import configure_jax

configure_jax("auto")

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from eamax.accumulators import solve_volterra_fpt  # noqa: E402
from eamax.flows.checkpoint import read_metadata  # noqa: E402
from flax import nnx  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy import stats  # noqa: E402

from confrdm_jax.flows_affine import (  # noqa: E402
    load_conditioner,
    make_mlp_conditioner,
    spline_flow,
    spline_knots,
)

RUN_TAG = "affine_log_deep_clip_100k"

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
OUTDIR = ROOT / "figures"

RDM_CONDITIONER = (
    OUTPUTS
    / "rdm/model.flow_affine=true/model.flow_log_inputs=true/model.flow_num_hidden=2"
    / "model.num_bins=12/model.num_mid=128"
    / "model.training_prior.b_max=3.5/model.training_prior.b_min=0.25"
    / "model.training_prior.s_max=3.5/model.training_prior.s_min=0.25"
    / "optimizer=adam_cosine_decay_clip/train_steps=100000/conditioner"
)
CRDM_CONDITIONER = (
    OUTPUTS
    / "crdm/model.flow_affine=true/model.flow_log_inputs=true/model.flow_num_hidden=2"
    / "model.num_bins=12/model.num_mid=128/model.sampler.dt=0.0005"
    / "optimizer=adam_cosine_decay_clip/train_steps=100000/conditioner"
)

Z_MIN, Z_MAX, NUM_Z = -5.5, 2.0, 1000
Z_RANGE = np.linspace(Z_MIN, Z_MAX, NUM_Z)
RT_RANGE = np.linspace(1e-4, 4.0, NUM_Z)

# The Volterra reference is solved once per parameter set on its own fine grid and
# interpolated onto RT_RANGE; past VOLTERRA_T_MAX (beyond the plotted range) it is NaN.
VOLTERRA_DT = 0.0005
VOLTERRA_T_MAX = 1.5

# Contexts are in each flow's input order: Wald (v, s, b); conflict Wald (v, amp, tau, s, b).
RDM_SETS = [
    {"context": (1.0, 1.0, 1.5), "label": r"$\nu{=}1$", "color": "#1b9e77"},
    {"context": (1.5, 1.0, 1.5), "label": r"$\nu{=}1.5$", "color": "#d95f02"},
    {"context": (2.5, 1.0, 1.5), "label": r"$\nu{=}2.5$", "color": "#7570b3"},
]
CRDM_SETS = [
    {"context": (4.0, 0.30, 0.15, 0.8, 0.9), "label": r"$\tau{=}0.15$", "color": "#1b9e77"},
    {"context": (4.0, 0.30, 0.10, 0.8, 0.9), "label": r"$\tau{=}0.10$", "color": "#d95f02"},
    {"context": (4.0, 0.30, 0.05, 0.8, 0.9), "label": r"$\tau{=}0.05$", "color": "#7570b3"},
]

LW = 1.8
ALPHA_FILL = 0.13


def load_from_sidecar(path):
    """Load a conditioner, building its template from the checkpoint's sidecar."""
    meta = read_metadata(path)
    if meta is None:
        raise FileNotFoundError(f"No conditioner sidecar at {path}")
    template = make_mlp_conditioner(
        nnx.Rngs(default=0),
        num_in=meta["num_in"],
        num_mid=meta["num_mid"],
        num_bins=meta["num_bins"],
        affine=meta.get("affine", False),
        spline_range=meta.get("spline_range", 5.0),
        boundary_slopes=meta.get("boundary_slopes", "identity"),
        input_scaling=meta.get("input_scaling"),
        num_hidden=meta.get("num_hidden", 1),
    )
    conditioner = load_conditioner(
        template, str(path), context_names=meta["context_names"]
    )
    conditioner.eval()
    return conditioner


def wald_density(v, s, b):
    """Analytic inverse Gaussian first-passage density on RT_RANGE."""
    mu, lam = b / v, (b / s) ** 2
    return stats.invgauss.pdf(RT_RANGE, mu / lam, scale=lam)


def volterra_density(v, amp, tau, s, b):
    """Volterra first-passage density on RT_RANGE, zero below the grid, NaN beyond it."""
    num_steps = round(VOLTERRA_T_MAX / VOLTERRA_DT)
    g, _ = solve_volterra_fpt(v, amp, tau, s, b, VOLTERRA_DT, num_steps=num_steps)
    grid = VOLTERRA_DT * np.arange(0, num_steps + 1)
    g = np.concatenate([[0.0], np.asarray(g)])
    return np.interp(RT_RANGE, grid, g, right=np.nan)


def flow_curves(conditioner, context):
    """The flow's density on RT_RANGE, its bijection on Z_RANGE, and its knots.

    The knots are also returned in time, with the flow's density evaluated there, so
    they can be marked on the density curve.
    """
    context = jnp.asarray(context)
    log_p, flow = spline_flow(jnp.asarray(RT_RANGE), context, conditioner)
    log_t = np.log(np.asarray(flow.bijector.forward(jnp.asarray(Z_RANGE))))
    z_knots, log_t_knots = spline_knots(conditioner, context)
    t_knots = jnp.exp(log_t_knots)
    log_p_knots, _ = spline_flow(t_knots, context, conditioner)
    return {
        "density": np.exp(np.asarray(log_p)),
        "log_t": log_t,
        "z_knots": np.asarray(z_knots),
        "log_t_knots": np.asarray(log_t_knots),
        "t_knots": np.asarray(t_knots),
        "density_knots": np.exp(np.asarray(log_p_knots)),
    }


def plot_bijection(ax, sets, curves, title):
    for ps, c in zip(sets, curves):
        visible = (c["z_knots"] >= Z_MIN) & (c["z_knots"] <= Z_MAX)
        ax.plot(Z_RANGE, c["log_t"], color=ps["color"], lw=LW, label=ps["label"])
        ax.scatter(
            c["z_knots"][visible],
            c["log_t_knots"][visible],
            color="white",
            edgecolors=ps["color"],
            linewidths=1.2,
            s=30,
            zorder=6,
        )
    ax.set_xlim(Z_MIN, Z_MAX)
    ax.set_xlabel("Base distribution z")
    ax.set_ylabel("log FPT (in s)")
    ax.set_title(title)


def plot_density(ax, sets, curves, references, reference_label, xlim=None):
    ax_z = ax.twiny()
    p_z = stats.norm.pdf(Z_RANGE)
    ax_z.plot(Z_RANGE, p_z, color="0.40", lw=LW, ls="--", label="Base N(0,1)")
    ax_z.fill_between(Z_RANGE, p_z, alpha=0.10, color="0.40")
    ax_z.set_xlim(Z_MIN, Z_MAX)
    ax_z.set_xlabel("Base distribution z", labelpad=5)
    ax_z.spines["top"].set_visible(True)

    for ps, c, ref in zip(sets, curves, references):
        ax.plot(RT_RANGE, c["density"], color=ps["color"], lw=LW, label=ps["label"])
        ax.plot(
            RT_RANGE, ref, color=ps["color"], lw=LW * 1.85, ls=":", alpha=0.85,
            label="_nolegend_",
        )
        ax.fill_between(RT_RANGE, c["density"], alpha=ALPHA_FILL, color=ps["color"])

    ax.set_ylim(bottom=0)
    if xlim is not None:
        ax.set_xlim(*xlim)

    t_min, t_max = ax.get_xlim()
    for ps, c in zip(sets, curves):
        visible = (c["t_knots"] >= t_min) & (c["t_knots"] <= t_max)
        ax.scatter(
            c["t_knots"][visible],
            c["density_knots"][visible],
            color="white",
            edgecolors=ps["color"],
            linewidths=1.2,
            s=30,
            zorder=6,
        )
    ax.set_xlabel("FPT (in s)")
    ax.set_ylabel("Density")

    base_handles, base_labels = ax_z.get_legend_handles_labels()
    flow_handles, flow_labels = ax.get_legend_handles_labels()
    reference_proxy = Line2D([0], [0], color="0.45", lw=LW * 0.85, ls=":")
    ax.legend(
        base_handles + flow_handles + [reference_proxy],
        base_labels + flow_labels + [reference_label],
        loc="upper right",
        frameon=False,
    )


def main():
    rdm_conditioner = load_from_sidecar(RDM_CONDITIONER)
    crdm_conditioner = load_from_sidecar(CRDM_CONDITIONER)

    rdm_curves = [flow_curves(rdm_conditioner, ps["context"]) for ps in RDM_SETS]
    rdm_refs = [wald_density(*ps["context"]) for ps in RDM_SETS]
    crdm_curves = [flow_curves(crdm_conditioner, ps["context"]) for ps in CRDM_SETS]
    crdm_refs = [volterra_density(*ps["context"]) for ps in CRDM_SETS]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    fig, ((ax_bij_rdm, ax_dens_rdm), (ax_bij_crdm, ax_dens_crdm)) = plt.subplots(
        2, 2, figsize=(12, 8)
    )

    plot_bijection(ax_bij_rdm, RDM_SETS, rdm_curves, "Wald")
    plot_density(ax_dens_rdm, RDM_SETS, rdm_curves, rdm_refs, "Analytic")
    plot_bijection(ax_bij_crdm, CRDM_SETS, crdm_curves, "Conflict Wald")
    plot_density(
        ax_dens_crdm, CRDM_SETS, crdm_curves, crdm_refs, "Numerical", xlim=(0, 1.0)
    )

    for ax, letter in zip([ax_bij_rdm, ax_dens_rdm, ax_bij_crdm, ax_dens_crdm], "ABCD"):
        ax.text(
            -0.10, 1.06, letter, transform=ax.transAxes,
            fontsize=18, fontweight="bold", va="top",
        )

    fig.tight_layout()
    OUTDIR.mkdir(exist_ok=True)
    out = OUTDIR / f"spline_flow_transformation_{RUN_TAG}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
