"""Where a run's results live, for the code that reads them outside Hydra.

A run is named by its ``run_tag``, which an experiment config sets -- the final run's is in
``conf_jax/experiment/final.yaml``, and that file is its only definition. Hydra writes a run
to ``outputs/<model>/<run_tag>/`` (sweeps to ``multirun/<model>/<run_tag>/``), with any
override beyond the experiment's as a subdirectory, e.g. ``model.sampler.dt=0.005`` or
``test_num_obs=50``.

This module reads that tag and rebuilds those paths for the figure scripts, the SLURM jobs
and ``conf_jax/compare_densities.yaml``, so none of them writes either down again. It holds
no values of its own.

From Python::

    from confrdm_jax import runs

    runs.conditioner_dir("crdm", dt=0.005)   # .../crdm/<tag>/model.sampler.dt=0.005/conditioner
    runs.run_dir("rdm", "test_num_obs=50", multirun=True)

From the shell::

    python -m confrdm_jax.runs run-tag
    python -m confrdm_jax.runs run-dir crdm model.sampler.dt=0.005
    python -m confrdm_jax.runs run-dir rdm --multirun
    python -m confrdm_jax.runs num-obs        # 50,250,500,1000, for a Hydra sweep

In a Hydra config, after importing this module::

    conditioner_path: ${run_dir:crdm,model.sampler.dt=0.005}/conditioner

The figure settings -- trial counts, convergence thresholds, palette -- come from
``conf_jax/figures.yaml`` through :func:`figure_settings`.
"""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

__all__ = [
    "CONF_DIR",
    "FINAL_EXPERIMENT",
    "ROOT",
    "conditioner_dir",
    "figure_settings",
    "run_dir",
    "run_tag",
]

#: The repository root. This package is used from an editable install of the repository.
ROOT = Path(__file__).resolve().parents[2]
CONF_DIR = ROOT / "conf_jax"
#: The experiment config that defines the final run.
FINAL_EXPERIMENT = CONF_DIR / "experiment" / "final.yaml"


def run_tag(experiment=FINAL_EXPERIMENT):
    """The ``run_tag`` an experiment config sets; the final run's by default."""
    return str(OmegaConf.load(experiment)["run_tag"])


def run_dir(model, *overrides, multirun=False, tag=None, root=ROOT):
    """A run's output directory.

    Parameters
    ----------
    model : str
        ``rdm`` or ``crdm``.
    *overrides : str
        Overrides beyond the experiment's, as passed to Hydra (``"model.sampler.dt=0.005"``).
        Joined as Hydra's ``override_dirname`` does: the ``key=value`` strings sorted and
        joined with ``/``.
    multirun : bool, optional
        A sweep (``--multirun``) rather than a single run.
    tag : str, optional
        Defaults to the final run's :func:`run_tag`.
    root : path, optional
        The directory ``outputs/`` and ``multirun/`` are under.

    Returns
    -------
    pathlib.Path
    """
    base = Path(root) / ("multirun" if multirun else "outputs") / model / (tag or run_tag())
    return base.joinpath(*sorted(overrides))


def conditioner_dir(model, dt=None, **kwargs):
    """A run's checkpoint directory, ``run_dir(...) / "conditioner"``.

    ``dt`` selects one of the CRDM flows trained at another simulator step: any value other
    than the model's configured ``sampler.dt`` (``conf_jax/model/crdm.yaml``) is the run's
    ``model.sampler.dt=<dt>`` subdirectory. Further keyword arguments go to :func:`run_dir`.
    """
    overrides = []
    if dt is not None:
        default = OmegaConf.load(CONF_DIR / "model" / f"{model}.yaml")["sampler"].get("dt")
        if default is None:
            raise ValueError(f"model {model!r} has no simulator step to select.")
        if float(dt) != float(default):
            overrides.append(f"model.sampler.dt={dt}")
    return run_dir(model, *overrides, **kwargs) / "conditioner"


def figure_settings():
    """``conf_jax/figures.yaml`` as a plain dict."""
    return OmegaConf.to_container(OmegaConf.load(CONF_DIR / "figures.yaml"), resolve=True)


def _run_dir_resolver(model, *overrides):
    return str(run_dir(model, *overrides))


# For `${run_dir:...}` in Hydra configs. Re-registering on reimport would raise.
if not OmegaConf.has_resolver("run_dir"):
    OmegaConf.register_new_resolver("run_dir", _run_dir_resolver)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m confrdm_jax.runs", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run-tag", help="print the final run's tag")
    directory = commands.add_parser("run-dir", help="print a run's output directory")
    directory.add_argument("model", choices=["rdm", "crdm"])
    directory.add_argument("overrides", nargs="*")
    directory.add_argument("--multirun", action="store_true")
    commands.add_parser("num-obs", help="print the recovery trial counts, comma-separated")
    args = parser.parse_args(argv)

    if args.command == "run-tag":
        print(run_tag())
    elif args.command == "run-dir":
        print(run_dir(args.model, *args.overrides, multirun=args.multirun))
    else:
        print(",".join(str(n) for n in figure_settings()["num_obs"]))


if __name__ == "__main__":
    main()
