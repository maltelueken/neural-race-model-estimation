"""Plot the hierarchical prior for the RDM and CRDM models.

Three quantities per parameter, one column each:

    exp(mu)  population location on the natural scale (the median of the
             lognormal, matching how ``posterior["mu"]`` is stored)
    s        between-subject SD, which stays in log space -- it is a standard
             deviation *of* log-parameters and so is dimensionless
    theta    subject-level parameters on the natural scale

The covariance factor ``psi_raw`` is deliberately not shown.

The dashed rule in the theta column is the edge of the box the neural flow was
trained over (``conf_jax/prior/wald_uniform.yaml`` /
``crdm_single_uniform.yaml``); outside it the flow extrapolates.  The x-limit is
stretched to keep that rule on screen, so the gap between the density and the
rule reads directly as headroom.  Parameters that are not themselves a flow
input (v_slope, t0) have no rule.

Everything is sampled from the live Hydra config, so the figure tracks
``conf_jax`` rather than restating it.

Output: ``figures/hierarchical_prior_{rdm,crdm}.png``

Run from anywhere::

    python scripts/create_figure_hierarchical_prior.py
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from hydra import compose
from hydra import initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "figures"

# mu and s get one draw per population, so the top two columns are 20x sparser
# than the subject-level one; this is set for their smoothness, not theta's.
NUM_POPULATIONS = 40_000
NUM_SUBJECTS = 20

# Categorical slots 1-3 of the reference palette, in fixed order.  Documented as
# clearing the all-pairs CVD and normal-vision floors in both modes, which is the
# case that applies here -- these are small multiples, not adjacent bars.
COLOR_MU, COLOR_S, COLOR_THETA = "#2a78d6", "#eb6834", "#1baf7a"

INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#8a8880"
GRID = "#e6e5e1"

# Which flow-box limit, if any, bounds each subject-level parameter.  The keys
# are parameter names, the values the `*_max` field of the training prior.
# v_intercept is v_c_false, which the flow does see; v_slope only ever enters as
# part of the sum v_intercept + v_slope, so it carries no limit of its own.
BOX_KEY = {
    "v_intercept": "v_max",
    "s_true": "s_max",
    "b": "b_max",
    "v_c_intercept": "v_c_max",
    "amp": "amp_max",
    "tau": "tau_max",
}

PRETTY = {
    "v_intercept": r"$v_{\mathrm{intercept}}$",
    "v_slope": r"$v_{\mathrm{slope}}$",
    "v_c_intercept": r"$v_{c,\mathrm{intercept}}$",
    "v_c_slope": r"$v_{c,\mathrm{slope}}$",
    "amp": r"$\mathrm{amp}$",
    "tau": r"$\tau$",
    "s_true": r"$s_{\mathrm{true}}$",
    "b": r"$b$",
    "t0": r"$t_0$",
}


def draw_prior(model):
    """Sample (exp(mu), s, theta) from a model's hierarchical prior.

    Returns:
        ``(mu, s, theta, param_names, box)`` -- `mu` and `s` are
        ``(NUM_POPULATIONS, P)``, `theta` is ``(NUM_POPULATIONS * NUM_SUBJECTS,
        P)`` on the natural scale, and `box` maps ``*_max`` names to the
        training prior's limits.
    """
    with initialize_config_dir(config_dir=str(ROOT / "conf_jax"), version_base=None):
        cfg = compose(config_name="config", overrides=[f"model={model}"])
        hier = cfg.model.hierarchical
        prior_cfg = OmegaConf.to_container(hier.prior, resolve=True)
        training_prior = OmegaConf.to_container(cfg.model.training_prior, resolve=True)
        prior = instantiate(hier.prior_factory)(NUM_SUBJECTS, **prior_cfg)
        param_names = list(hier.param_names)

    num_params_ncp = len(param_names) - 2  # b and t0 are the centered block

    def one_population(key):
        params = prior.sample(seed=key)
        chol = params["s"][:, None] * params["psi_raw"]
        theta_ncp = params["mu"][:num_params_ncp] + jnp.einsum(
            "nj,ij->ni", params["z"], chol[:num_params_ncp, :num_params_ncp],
        )
        log_theta = jnp.concatenate([theta_ncp, params["theta_bt"]], axis=-1)
        return params["mu"], params["s"], jnp.exp(log_theta)

    keys = jax.random.split(jax.random.key(0), NUM_POPULATIONS)
    mu, s, theta = jax.vmap(one_population)(keys)

    box = {k: v for k, v in training_prior.items() if k.endswith("_max")}
    return (
        np.exp(np.asarray(mu)),
        np.asarray(s),
        np.asarray(theta).reshape(-1, len(param_names)),
        param_names,
        box,
    )


# A limit further out than this multiple of the density's own extent is shown as
# an off-scale note instead of a rule: stretching the axis to reach it would
# squash the distribution into the left margin and cost more than the headroom
# it illustrates.
MAX_LIMIT_STRETCH = 2.2


def density_panel(ax, values, color, limit=None, zero_anchor=False):
    """Filled density for one quantity, with the flow-box rule if there is one."""
    extent = np.percentile(values, 99.9)
    lower = 0.0 if zero_anchor else np.percentile(values, 0.1)

    on_scale = limit is not None and limit <= extent * MAX_LIMIT_STRETCH
    upper = max(extent, limit * 1.04) if on_scale else extent

    ax.hist(
        np.clip(values, lower, upper),
        bins=80, range=(lower, upper), density=True,
        color=color, alpha=0.85, edgecolor="none",
    )

    if on_scale:
        ax.axvline(limit, color=INK_MUTED, lw=1.2, ls=(0, (4, 3)), zorder=3)
        ax.annotate(
            f"flow limit {limit:g}",
            xy=(limit, 0.94), xycoords=("data", "axes fraction"),
            xytext=(-5, 0), textcoords="offset points",
            ha="right", va="top", fontsize=7.5, color=INK_MUTED, rotation=90,
        )
    elif limit is not None:
        ax.annotate(
            f"flow limit {limit:g} →",
            xy=(0.995, 0.88), xycoords="axes fraction",
            ha="right", va="top", fontsize=7.5, color=INK_MUTED,
        )

    ax.set_xlim(lower, upper)
    ax.set_yticks([])
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=INK_SECONDARY, labelsize=8, length=3, color=GRID)


def make_figure(model, path):
    mu, s, theta, param_names, box = draw_prior(model)
    num_params = len(param_names)

    # The title block needs a fixed height in inches, not a fraction: the figure
    # grows with the parameter count, so a fractional top margin would leave the
    # 7-row CRDM headings crowded and the 5-row RDM ones adrift.
    title_block_in = 1.15
    fig_height = 1.42 * num_params + title_block_in + 0.45

    fig, axes = plt.subplots(
        num_params, 3, figsize=(9.2, fig_height),
        gridspec_kw={"hspace": 0.62, "wspace": 0.2},
    )
    axes = np.atleast_2d(axes)

    columns = [
        (r"population location  $\exp(\mu)$", mu, COLOR_MU),
        (r"between-subject SD  $s$  (log scale)", s, COLOR_S),
        (r"subject-level  $\theta$", theta, COLOR_THETA),
    ]

    for row, name in enumerate(param_names):
        for col, (_, data, color) in enumerate(columns):
            # Only the subject-level column is bounded by the flow's box: mu is a
            # population location and s is a log-space SD, neither of which the
            # conditioner ever takes as input.
            limit = box.get(BOX_KEY.get(name)) if col == 2 else None
            density_panel(
                axes[row, col], data[:, row], color, limit, zero_anchor=(col == 2),
            )

        axes[row, 0].set_ylabel(
            PRETTY.get(name, name), rotation=0, ha="right", va="center",
            fontsize=13, color=INK_PRIMARY, labelpad=14,
        )

    for col, (heading, _, color) in enumerate(columns):
        axes[0, col].set_title(
            heading, fontsize=10.5, color=INK_PRIMARY, pad=14, loc="left",
        )
        # A short colored rule under each heading ties the column to its fill
        # without putting the series color on the text itself.
        axes[0, col].annotate(
            "", xy=(0.0, 1.07), xytext=(0.18, 1.07), xycoords="axes fraction",
            arrowprops=dict(arrowstyle="-", color=color, lw=2.5),
        )

    fig.text(
        0.055, 1 - 0.30 / fig_height, f"{model.upper()} hierarchical prior",
        ha="left", va="top", fontsize=15, color=INK_PRIMARY, weight="semibold",
    )
    fig.text(
        0.055, 1 - 0.56 / fig_height,
        f"{NUM_POPULATIONS:,} populations × {NUM_SUBJECTS} subjects sampled from conf_jax; "
        "covariance factor not shown",
        ha="left", va="top", fontsize=9, color=INK_SECONDARY,
    )

    fig.subplots_adjust(
        left=0.115, right=0.985,
        top=1 - title_block_in / fig_height, bottom=0.35 / fig_height,
    )
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    OUTDIR.mkdir(exist_ok=True)
    mpl.rcParams.update({"font.family": "sans-serif", "axes.unicode_minus": False})
    for model in ("rdm", "crdm"):
        make_figure(model, OUTDIR / f"hierarchical_prior_{model}.png")
