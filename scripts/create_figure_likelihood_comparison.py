"""Neural vs. reference race densities for the RDM and CRDM.

Script version of ``notebooks/create_figure_likelihood_comparison.ipynb``, pointed at the flows
trained for 100k steps with an affine layer, two hidden layers, log-scaled inputs and
gradient clipping (``optimizer=adam_cosine_decay_clip``) -- the same runs as
``create_figures_parameter_recovery_hierarchical.py``.

Each panel shows the joint density of (response time, choice) for one parameter set, with
choice 0 (non-target) mirrored onto negative response times:

  * RDM row: the neural likelihood against the analytic inverse Gaussian race.
  * CRDM rows, one per training step size ``dt`` in ``CRDM_DTS``: the hybrid neural
    likelihood of the flow trained at that ``dt`` against the same race with the Volterra
    first-passage-time solver on the pulsed accumulator. The solver always runs at
    ``REFERENCE_DT``, so every flow is compared with the same, fine-grained density; at
    coarse ``dt`` the solver itself loses mass for fast parameter sets. Congruent and
    incongruent trials are collapsed into their equal-weight mixture.

The conditioner template is rebuilt from the checkpoint's sidecar (depth, affine layout,
spline settings, log-input scaling), so the script cannot load a checkpoint into the
wrong architecture.

Output: ``figures/likelihood_comparison_{RUN_TAG}.png``

Run from anywhere:
    python scripts/create_figure_likelihood_comparison.py
"""

from pathlib import Path

from confrdm_jax import configure_jax

configure_jax("auto")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from eamax.accumulators import VolterraPulsedWald, Wald  # noqa: E402
from eamax.accumulators.volterra import solve_volterra_fpt  # noqa: E402
from eamax.flows.checkpoint import read_metadata  # noqa: E402
from flax import nnx  # noqa: E402

from confrdm_jax.flows_affine import load_conditioner, make_mlp_conditioner  # noqa: E402
from confrdm_jax.likelihoods import (  # noqa: E402
    _hybrid_race,
    create_crdm_likelihood_factory_approx,
    create_rdm_likelihood_factory_approx,
    create_rdm_two_accumulators_likelihood,
)
from confrdm_jax.specs import crdm_spec, make_design  # noqa: E402

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
# One CRDM flow per training-simulator step size, all compared against the Volterra solver
# at REFERENCE_DT.
CRDM_DTS = [0.05, 0.005, 0.0005]
REFERENCE_DT = 0.0005
CRDM_CONDITIONERS = {
    dt: OUTPUTS
    / "crdm/model.flow_affine=true/model.flow_log_inputs=true/model.flow_num_hidden=2"
    / f"model.num_bins=12/model.num_mid=128/model.sampler.dt={dt}"
    / "optimizer=adam_cosine_decay_clip/train_steps=100000/conditioner"
    for dt in CRDM_DTS
}

DT_TEST = 0.005
T_MAX = 4.0

# [v_intercept, v_slope, s_true, b, t0]
RDM_PARAMS = np.array(
    [
        [1.0, 1.0, 1.2, 1.5, 0.3],
        [1.0, 1.5, 1.2, 1.5, 0.3],
        [1.0, 2.0, 1.2, 1.5, 0.3],
    ]
)

# [v_c_intercept, v_c_slope, amp, tau, s_true, b, t0]
CRDM_PARAMS = np.array(
    [
        [1.0, 4.0, 0.3, 0.15, 0.8, 0.9, 0.3],
        [1.0, 4.0, 0.3, 0.10, 0.8, 0.9, 0.3],
        [1.0, 4.0, 0.3, 0.05, 0.8, 0.9, 0.3],
    ]
)

CONDITIONS = {"congruent": 1.0, "incongruent": 0.0}

COLORS = {"Analytic": "#7570b3", "Numerical": "#1b9e77", "Neural": "#d95f02"}


class SolveOnceVolterra(VolterraPulsedWald):
    """`VolterraPulsedWald` that solves once per *distinct* parameter set, not per trial.

    The eamax accumulator vmaps the solver over trials, materialising one
    ``num_steps x num_steps`` kernel each: at ``dt = 5e-4`` that is 512 MB per trial. On
    this grid every trial shares one parameter set, so the solve is done once and the
    trials are interpolated from it. Needs concrete (non-traced) parameters, so the
    likelihood using it must not be jitted.
    """

    def log_pdf_sf(self, t, params):
        t = np.asarray(t)
        rows = np.stack(
            [
                np.ravel(np.broadcast_to(np.asarray(params[name]), t.shape))
                for name in self.param_names
            ],
            axis=1,
        )
        rows[:, 2:] = np.maximum(rows[:, 2:], self.min_param)  # tau, s, b
        unique, inverse = np.unique(rows, axis=0, return_inverse=True)
        # The solver's grid starts at dt. eamax interpolates below it by holding g(dt), which
        # at dt = 0.05 puts a visible plateau of density on rt < t0. Anchoring the grid at
        # (0, 0) instead makes the density 0 for non-positive decision times and linear
        # up to dt; beyond the end it still holds the endpoint, as in eamax.
        grid = np.arange(0, self.num_steps + 1) * self.dt

        log_pdf = np.empty(t.size)
        log_sf = np.empty(t.size)
        for k, row in enumerate(unique):
            g, cdf = solve_volterra_fpt(
                *row, self.dt, num_steps=self.num_steps, a_shape=self.a_shape
            )
            g = np.concatenate([[0.0], np.asarray(g)])
            cdf = np.concatenate([[0.0], np.asarray(cdf)])
            idx = inverse.ravel() == k
            decision_time = t.ravel()[idx]
            log_pdf[idx] = np.log(np.maximum(np.interp(decision_time, grid, g), 1e-30))
            log_sf[idx] = np.log(
                np.maximum(1.0 - np.interp(decision_time, grid, cdf), 1e-30)
            )
        return jnp.asarray(log_pdf.reshape(t.shape)), jnp.asarray(log_sf.reshape(t.shape))


def create_crdm_likelihood_volterra(data, dt, t_max):
    """`confrdm_jax.likelihoods.create_crdm_likelihood_volterra` with `SolveOnceVolterra`."""
    spec = crdm_spec()
    design = make_design(data)
    pulsed = SolveOnceVolterra(dt=dt, t_max=t_max)
    plain = Wald()
    return lambda theta: _hybrid_race(spec, pulsed, plain, theta, design)


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


def make_grid(condition=None):
    """Every (rt, choice) on the test grid, optionally with a constant condition column."""
    rt = jnp.arange(DT_TEST, T_MAX + DT_TEST, DT_TEST)
    choice = jnp.array([0.0, 1.0])
    columns = [jnp.tile(rt, choice.shape[0]), jnp.repeat(choice, rt.shape[0])]
    if condition is not None:
        columns.append(jnp.full(columns[0].shape, condition))
    return jnp.stack(columns, axis=1)


def densities(likelihood_fun, params):
    """Per-trial densities, one row per parameter set (parameters passed in log space)."""
    return np.stack([np.exp(likelihood_fun(jnp.log(jnp.asarray(p)))) for p in params])


def signed_rt(data):
    """Response time with choice 0 mirrored to negative values, and its sort order."""
    x = np.where(np.asarray(data[:, 1]) == 0, -1.0, 1.0) * np.asarray(data[:, 0])
    return x, np.argsort(x)


def compute():
    rdm_conditioner = load_from_sidecar(RDM_CONDITIONER)

    rows = []

    data = make_grid()
    rows.append(
        {
            "name": "rdm",
            "label": "RDM",
            "data": data,
            "reference_label": "Analytic",
            "reference": densities(create_rdm_two_accumulators_likelihood(data), RDM_PARAMS),
            "neural": densities(
                jax.jit(create_rdm_likelihood_factory_approx(rdm_conditioner)(data)),
                RDM_PARAMS,
            ),
            "titles": [rf"$\nu_\text{{slope}} = {p[1]}$" for p in RDM_PARAMS],
            "xlim": (-3, 3),
            "ylim": (-0.25, 2.0),
        }
    )

    # The reference does not depend on the flow, so it is solved once for every row.
    grids = {name: make_grid(condition) for name, condition in CONDITIONS.items()}
    reference = [
        densities(
            create_crdm_likelihood_volterra(data, dt=REFERENCE_DT, t_max=T_MAX), CRDM_PARAMS
        )
        for data in grids.values()
    ]

    for dt in CRDM_DTS:
        crdm_conditioner = load_from_sidecar(CRDM_CONDITIONERS[dt])
        neural = [
            densities(
                jax.jit(create_crdm_likelihood_factory_approx(crdm_conditioner)(data)),
                CRDM_PARAMS,
            )
            for data in grids.values()
        ]
        # Congruent and incongruent collapsed: the equal-weight mixture, i.e. the marginal
        # density of (rt, choice) in a design with as many trials of each.
        rows.append(
            {
                "name": f"crdm dt={dt}",
                "label": f"CRDM, $dt$ = {dt}",
                "data": grids["congruent"],
                "reference_label": "Numerical",
                "reference": np.mean(reference, axis=0),
                "neural": np.mean(neural, axis=0),
                "titles": [rf"$\tau = {p[3]}$" for p in CRDM_PARAMS],
                "xlim": (-1, 1),
                "ylim": (-0.5, 8),
            }
        )

    return rows


def report(rows):
    """Max absolute and total-variation discrepancy between neural and reference."""
    print(f"{'row':>30} {'set':>4} {'max |diff|':>11} {'TV':>8}")
    for row in rows:
        for i, (ref, nn) in enumerate(zip(row["reference"], row["neural"])):
            diff = np.abs(ref - nn)
            tv = 0.5 * diff.sum() * DT_TEST
            print(f"{row['name']:>30} {i:>4} {diff.max():>11.4f} {tv:>8.4f}")


def plot(rows):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 10,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, axarr = plt.subplots(len(rows), 3, figsize=(10, 2.5 * len(rows)))
    handles = {}

    for r, (row, ax_row) in enumerate(zip(rows, axarr)):
        x, order = signed_rt(row["data"])
        for i, ax in enumerate(ax_row):
            label = row["reference_label"]
            (h_ref,) = ax.plot(
                x[order], row["reference"][i][order], alpha=0.7, color=COLORS[label]
            )
            (h_nn,) = ax.plot(
                x[order], row["neural"][i][order], "--", alpha=0.7, color=COLORS["Neural"]
            )
            handles.setdefault(label, h_ref)
            handles.setdefault("Neural", h_nn)
            ax.set_xlim(row["xlim"])
            ax.set_ylim(row["ylim"])
            ax.set_title(row["titles"][i])
            if r == len(rows) - 1:
                ax.set_xlabel("Response time (in s)")
            if i == 0:
                ax.set_ylabel("Density")
        ax_row[-1].annotate(
            row["label"],
            xy=(1.04, 0.5),
            xycoords="axes fraction",
            rotation=270,
            va="center",
            ha="left",
        )

    fig.legend(
        handles=list(handles.values()),
        labels=list(handles.keys()),
        loc="upper center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.tight_layout()

    out = OUTDIR / f"likelihood_comparison_{RUN_TAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = compute()
    report(rows)
    plot(rows)


if __name__ == "__main__":
    main()
