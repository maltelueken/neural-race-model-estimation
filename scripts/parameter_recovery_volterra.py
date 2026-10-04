"""Single-subject CRDM recovery with the Volterra likelihood, on the flow recovery's data sets.

The reference counterpart of ``scripts/parameter_recovery.py`` for the CRDM, which has no
closed form: the same data sets, the same recovery prior, the same NUTS settings
(``mcmc.num_warmup`` / ``num_sampling`` / ``num_chains``, ``init.*``) and the same starting
positions, with the flow replaced by
:func:`confrdm_jax.likelihoods.create_crdm_likelihood_volterra` at grid step ``--dt``.

**Same data, same starts.** The data sets are read from the flow run's
``parameter_recovery_approx.nc`` rather than re-simulated, because the flow job simulated
them on a GPU and a CPU re-simulation can resolve a near-threshold crossing differently.
Each chain starts where the matching flow chain started: the starting positions are drawn
with the flow fit's per-data-set key, from the same prior and the same ``T0Support``.

**One work item is one chain of one data set.** The flow recovery fits every data set in one
XLA computation on a GPU; the Volterra solve is a sequential recursion that neither a GPU
nor extra CPU threads speed up much, so the parallelism is across chains instead. Each
item runs single-threaded and writes ``chains/d<dataset>_c<chain>.npz``; an item whose file
exists is skipped, so a job that ran out of time is resumed by resubmitting it. Splitting
the chains also removes the lockstep a vmapped run has, where every iteration waits for the
chain with the deepest tree. Every chain still adapts on its own, as
:func:`eamax.inference.warmup.window_adaptation` does for the flow fit.

**Cost** is set by the grid: the solver horizon is each data set's longest RT, so a solve is
``O((max_rt / dt)^2)`` and the number of trials barely matters. ``list`` orders the work
items longest grid first, which is what packs a node best.

Usage (``slurm/parameter_recovery_crdm_volterra.sh`` runs ``list`` then ``run`` per item)::

    python scripts/parameter_recovery_volterra.py list  --num-obs 500 [--dt 0.0005]
    python scripts/parameter_recovery_volterra.py run   --num-obs 500 --dataset 17 --chain 2
    python scripts/parameter_recovery_volterra.py merge --num-obs 500

Reads and writes ``multirun/crdm/<run_tag>/test_num_obs=<N>/`` (the flow recovery's sweep
directory): chains go to ``volterra_dt=<dt>/chains/``, and ``merge`` writes
``parameter_recovery_volterra_dt<dt>.nc`` beside ``parameter_recovery_approx.nc``, in the
same layout, holding only the data sets whose chains have all finished. Its ``subject``
coordinate keeps the original data-set indices, so it lines up with the flow file by label.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("parameter_recovery_volterra")

#: Flow recovery output the data sets and prior draws are read from.
APPROX_FILE = "parameter_recovery_approx.nc"


def recovery_dir(num_obs):
    """The flow recovery's sweep directory for `num_obs` trials per data set."""
    from confrdm_jax import runs

    return runs.run_dir("crdm", f"test_num_obs={num_obs}", multirun=True)


def chain_dir(num_obs, dt):
    return recovery_dir(num_obs) / f"volterra_dt={dt:g}" / "chains"


def chain_file(num_obs, dt, dataset, chain):
    return chain_dir(num_obs, dt) / f"d{dataset:03d}_c{chain}.npz"


def compose_config(num_obs):
    """The flow recovery's Hydra config: ``model=crdm +experiment=final test_num_obs=N``."""
    from hydra import compose, initialize_config_dir

    from confrdm_jax import runs

    with initialize_config_dir(config_dir=str(runs.CONF_DIR), version_base=None):
        return compose(
            config_name="config",
            overrides=["model=crdm", "+experiment=final", f"test_num_obs={num_obs}"],
        )


def load_recovery(num_obs):
    """The flow recovery's DataTree, with its data sets and true parameters."""
    import xarray as xr

    path = recovery_dir(num_obs) / APPROX_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist; the Volterra recovery reuses the flow recovery's data "
            "sets, so run slurm/parameter_recovery_crdm_single.sh first.",
        )
    return xr.open_datatree(path)


def observed_data(tree):
    """``(num_datasets, num_trials, 3)`` with columns ``[rt, choice, condition]``."""
    from confrdm_jax.specs import DATA_COL_NAMES

    observed = tree["observed_data"]
    return np.stack([observed[name].values for name in DATA_COL_NAMES], axis=-1)


# --------------------------------------------------------------------------------------- #
# list
# --------------------------------------------------------------------------------------- #


def cmd_list(args):
    """Print the unfinished work items as ``num_obs dataset chain``, longest grid first."""
    items = []
    for num_obs in args.num_obs:
        tree = load_recovery(num_obs)
        rt = tree["observed_data"]["rt"].values
        num_chains = tree["posterior"].sizes["chain"]
        grid = np.ceil(np.where(rt > 0, rt, 0.0).max(axis=1) / args.dt)
        for dataset in range(rt.shape[0]):
            for chain in range(num_chains):
                if not chain_file(num_obs, args.dt, dataset, chain).exists():
                    items.append((grid[dataset], num_obs, dataset, chain))
    for _, num_obs, dataset, chain in sorted(items, key=lambda item: -item[0]):
        print(num_obs, dataset, chain)


# --------------------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------------------- #


def cmd_run(args):
    """Fit one chain of one data set and write its draws."""
    out = chain_file(args.num_obs, args.dt, args.dataset, args.chain)
    if out.exists() and not args.overwrite:
        logger.info("%s exists; skipping", out)
        return

    from confrdm_jax import configure_jax

    configure_jax(args.device)

    import blackjax
    import jax
    import jax.numpy as jnp
    from eamax.inference import (
        T0Support,
        init_positions_from_prior,
        inference_loop_multiple_chains,
        min_valid_rt,
        window_adaptation,
    )
    from hydra.utils import instantiate

    from confrdm_jax.likelihoods import create_crdm_likelihood_volterra
    from confrdm_jax.priors import log_prior_fn
    from confrdm_jax.specs import CRDM_PARAM_NAMES, crdm_spec

    cfg = compose_config(args.num_obs)
    model_cfg = cfg["model"]
    num_datasets = cfg["test_num_datasets"]
    num_chains = cfg["mcmc"]["num_chains"]
    if not 0 <= args.chain < num_chains:
        raise ValueError(f"--chain must be in [0, {num_chains}), got {args.chain}")

    tree = load_recovery(args.num_obs)
    if tree["observed_data"].sizes["subject"] != num_datasets:
        raise ValueError(
            f"{APPROX_FILE} holds {tree['observed_data'].sizes['subject']} data sets, the "
            f"config says test_num_datasets={num_datasets}; the per-data-set keys depend on it.",
        )
    data = jnp.asarray(observed_data(tree)[args.dataset])
    truth = tree["constant_data"]["theta"].values[args.dataset]

    spec = crdm_spec()
    prior = instantiate(model_cfg["recovery_prior"])
    log_prior = log_prior_fn(prior)
    loglik = create_crdm_likelihood_volterra(
        data, dt=args.dt, censor_t_max=model_cfg["test_sampler"]["t_max"],
    )

    def logdensity_fun(x):
        return log_prior(x) + jnp.sum(loglik(x))

    # The flow fit's keys, exactly as scripts/parameter_recovery.py derives them: the
    # approximate run's per-data-set key, split into init / warm-up / sampling.
    test_key = jax.random.key(cfg["test_seed"] + cfg["test_num_obs"])
    _, sampling_key, _ = jax.random.split(test_key, 3)
    sampling_key_approx, _ = jax.random.split(sampling_key, 2)
    dataset_key = jax.random.split(sampling_key_approx, num_datasets)[args.dataset]
    init_key, warmup_key, loop_key = jax.random.split(dataset_key, 3)

    def draw_from_prior(key):
        return jnp.log(jnp.stack(prior.sample(seed=key)))

    support = T0Support.from_spec(
        spec, min_valid_rt(data[:, 0]), max_fraction=cfg["init"]["t0_max_fraction"],
    )
    init_positions, num_exhausted = init_positions_from_prior(
        draw_from_prior, num_chains, init_key,
        support=support, max_attempts=cfg["init"]["t0_rejection_max_attempts"],
    )
    # This chain's start, as a one-chain batch. Warm-up and sampling keys are this chain's
    # share of the flow fit's; they need not reproduce its draws, only be independent.
    start = init_positions[args.chain : args.chain + 1]
    chain_warmup_key = jax.random.split(warmup_key, num_chains)[args.chain]
    chain_loop_key = jax.random.split(loop_key, num_chains)[args.chain]

    rt = np.asarray(data[:, 0])
    logger.info(
        "num_obs=%d data set %d chain %d: dt=%g, solver grid %d steps (max rt %.3f s), "
        "truth %s",
        args.num_obs, args.dataset, args.chain, args.dt,
        int(np.ceil(rt[rt > 0].max() / args.dt - 1e-9)), rt[rt > 0].max(),
        dict(zip(CRDM_PARAM_NAMES, np.round(truth, 4), strict=True)),
    )

    timer = time.time()
    value, _ = jax.jit(jax.value_and_grad(logdensity_fun))(start[0])
    logger.info("log density at the start %.3f (compiled in %.1f s)", value, time.time() - timer)

    timer = time.time()
    last_state, kernel_params = window_adaptation(
        blackjax.nuts, logdensity_fun, start, cfg["mcmc"]["num_warmup"], chain_warmup_key,
        num_chains=1,
    )
    jax.block_until_ready(last_state.position)
    warmup_seconds = time.time() - timer
    step_size = float(kernel_params["step_size"][0])
    logger.info(
        "warm-up: %d steps in %.2f h, step size %.4g",
        cfg["mcmc"]["num_warmup"], warmup_seconds / 3600, step_size,
    )

    nuts_kernel = blackjax.nuts.build_kernel()

    def kernel(key, state, params):
        return nuts_kernel(key, state, logdensity_fun, **params)

    timer = time.time()
    positions, infos = inference_loop_multiple_chains(
        chain_loop_key, kernel, last_state, cfg["mcmc"]["num_sampling"], 1, kernel_params,
    )
    jax.block_until_ready(positions)
    sampling_seconds = time.time() - timer
    steps = np.asarray(infos.num_integration_steps)[:, 0]
    divergent = np.asarray(infos.is_divergent)[:, 0]
    logger.info(
        "sampling: %d draws in %.2f h, %.1f leapfrog steps per draw, %d divergences",
        cfg["mcmc"]["num_sampling"], sampling_seconds / 3600, steps.mean(), divergent.sum(),
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    # Written to a temporary name and renamed, so a job killed mid-write leaves no file
    # that `list` would mistake for a finished chain.
    partial = out.with_name(out.stem + ".partial.npz")
    np.savez(
        partial,
        draws=np.asarray(spec.constrain(positions))[:, 0],  # (num_sampling, P), natural scale
        step_size=step_size,
        num_exhausted=int(num_exhausted),
        num_integration_steps=steps,
        is_divergent=divergent,
        acceptance_rate=np.asarray(infos.acceptance_rate)[:, 0],
        warmup_seconds=warmup_seconds,
        sampling_seconds=sampling_seconds,
        dt=args.dt,
    )
    partial.rename(out)
    logger.info("wrote %s", out)


# --------------------------------------------------------------------------------------- #
# merge
# --------------------------------------------------------------------------------------- #


def cmd_merge(args):
    """Assemble finished chains into a netCDF in the flow recovery's layout."""
    import xarray as xr

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from parameter_recovery import _build_recovery_datatree

    from confrdm_jax.specs import CRDM_PARAM_NAMES

    for num_obs in args.num_obs:
        tree = load_recovery(num_obs)
        data = observed_data(tree)
        truth = tree["constant_data"]["theta"].values
        num_chains = tree["posterior"].sizes["chain"]

        complete = [
            dataset for dataset in range(data.shape[0])
            if all(chain_file(num_obs, args.dt, dataset, c).exists() for c in range(num_chains))
        ]
        if not complete:
            logger.warning("num_obs=%d: no data set has all %d chains yet", num_obs, num_chains)
            continue

        pieces = [
            [np.load(chain_file(num_obs, args.dt, dataset, c)) for c in range(num_chains)]
            for dataset in complete
        ]
        # (datasets, draws, chains, params), the layout _build_recovery_datatree expects.
        samples = np.stack([np.stack([p["draws"] for p in row], axis=1) for row in pieces])
        step_sizes = np.array([[float(p["step_size"]) for p in row] for row in pieces])
        num_divergent = np.array([[int(p["is_divergent"].sum()) for p in row] for row in pieces])
        hours = np.array([
            [(float(p["warmup_seconds"]) + float(p["sampling_seconds"])) / 3600 for p in row]
            for row in pieces
        ])

        merged = _build_recovery_datatree(
            samples=samples,
            data=data[complete],
            true_params=truth[complete],
            prior_ds=tree["prior"].to_dataset(),
            param_names=list(CRDM_PARAM_NAMES),
            step_sizes=step_sizes,
            attrs={
                "likelihood": "volterra",
                "volterra_dt": args.dt,
                # Starts are drawn per data set, so every chain of one carries the same count.
                "num_init_exhausted": int(sum(int(row[0]["num_exhausted"]) for row in pieces)),
            },
        )
        # _build_recovery_datatree numbers subjects 0..n-1; relabel with the data-set indices
        # so a subset lines up with the flow file by `sel`, and record the per-chain cost.
        groups = {
            name: (
                merged[name].to_dataset().assign_coords(subject=complete)
                if "subject" in merged[name].dims else merged[name].to_dataset()
            )
            for name in merged.children
        }
        groups["sample_stats"] = groups["sample_stats"].assign(
            num_divergent=(["subject", "chain"], num_divergent),
            hours=(["subject", "chain"], hours),
        )
        relabelled = xr.DataTree.from_dict(groups)
        relabelled.attrs.update(merged.attrs)

        path = recovery_dir(num_obs) / f"parameter_recovery_volterra_dt{args.dt:g}.nc"
        relabelled.to_netcdf(path)
        logger.info(
            "num_obs=%d: %d of %d data sets complete, wrote %s (chain hours: median %.1f, "
            "max %.1f)",
            num_obs, len(complete), data.shape[0], path, np.median(hours), hours.max(),
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def num_obs_list(text):
        return [int(n) for n in text.split(",")]

    for name in ("list", "merge"):
        sub = commands.add_parser(name)
        sub.add_argument("--num-obs", type=num_obs_list, required=True,
                         help="comma-separated trial counts, e.g. 50,250,500,1000")
        sub.add_argument("--dt", type=float, default=0.0005)

    run = commands.add_parser("run")
    run.add_argument("--num-obs", type=int, required=True)
    run.add_argument("--dataset", type=int, required=True)
    run.add_argument("--chain", type=int, required=True)
    run.add_argument("--dt", type=float, default=0.0005)
    run.add_argument("--device", default="cpu", help="passed to confrdm_jax.configure_jax")
    run.add_argument("--overwrite", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format=f"%(asctime)s [{os.getpid()}] %(message)s", stream=sys.stdout,
    )
    logging.getLogger("absl").setLevel(logging.ERROR)
    {"list": cmd_list, "run": cmd_run, "merge": cmd_merge}[args.command](args)


if __name__ == "__main__":
    main()
