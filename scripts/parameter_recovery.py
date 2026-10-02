"""Single-subject parameter recovery for a trained conditioner.

Simulates independent data sets from the recovery prior, fits each one with NUTS, and writes
the posteriors to netCDF for the notebooks to analyse. Each data set is fit independently —
they are stored under a ``subject`` coordinate for ArviZ's benefit, but nothing is shared
between them. For the genuinely multi-subject model see
``parameter_recovery_hierarchical.py``.

When the model has an analytic likelihood (RDM, ``run_reference_recovery: true``) the same
data are fit a second time with it, giving a reference posterior that
``scripts/c2st_recovery.py`` can compare the neural one against. The CRDM has no closed
form, so only the approximate fit runs.

Every data set is fit in one XLA computation: ``vmap`` over data sets, ``scan`` over draws,
``vmap`` over chains inside each step. That is what ``device=gpu`` buys — the whole study is
one job rather than one job per data set.

**Starting values** come from the prior, conditioned on ``t0 < 0.97 * min(rt)``
(:class:`eamax.inference.init.T0Support`), one independent draw per chain; every chain is
then adapted on its own by :func:`eamax.inference.warmup.window_adaptation` and sampled with
the tuning that comes back, unmodified. A collapsed step size is reported, not repaired —
see :data:`DEGENERATE_STEP_SIZE`.

Must be launched with the same overrides that produced the checkpoint — nothing in the
weights records the architecture::

    python scripts/parameter_recovery.py model=rdm

Outputs ``parameter_recovery_approx.nc`` and, when enabled, ``parameter_recovery_ref.nc`` in
the Hydra run directory.
"""

import logging
from pathlib import Path

import arviz as az
import blackjax
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from eamax.inference import (
    T0Support,
    init_positions_from_prior,
    inference_loop_multiple_chains,
    min_valid_rt,
    window_adaptation,
)
from flax import nnx
from hydra.utils import instantiate

from confrdm_jax import configure_jax
from confrdm_jax.flows_affine import flow_options, load_conditioner, make_mlp_conditioner
from confrdm_jax.priors import log_prior_fn
from confrdm_jax.specs import DATA_COL_NAMES, spec_for

logger = logging.getLogger(__name__)

logging.getLogger("absl").setLevel(logging.ERROR)

#: A step size this far below its siblings' is a collapsed window adaptation, not a
#: converged one: healthy chains on these posteriors land at 0.06-0.16. It is **reported and
#: not repaired**. Substituting the healthy chains' median was tried and measured: repaired
#: chains came out under-dispersed at 0.16-0.83x their siblings' spread and still broke R-hat
#: (1.94 / 2.60 / 2.76), falling to ~1.01 only once they were dropped. The cause is almost
#: always a start outside the ``t0`` support, which ``T0Support`` removes at the source.
DEGENERATE_STEP_SIZE = 1e-4


def _log_init_exhaustion(label, exhausted, num_datasets, num_chains):
    """Warn if any chain's `t0` rejection budget ran out and it was clipped.

    Exhaustion means the constraint and the prior disagree badly enough that rejection cannot
    bridge them, and the fallback is a clip — a point mass in the one coordinate the
    dispersion exists to spread. It should be 0.
    """
    total = int(jnp.sum(exhausted))
    if total:
        logger.warning(
            "%s: t0 rejection sampling exhausted for %d of %d chain starts; "
            "those fell back to clipping",
            label, total, num_datasets * num_chains,
        )


def _log_step_sizes(label, step_sizes):
    """Report the adapted step sizes and flag collapsed adaptations.

    Returns
    -------
    int
        Number of (data set, chain) pairs whose adaptation collapsed. Recorded on the output
        file so a bad fit is visible from the artifact, not only from the log.
    """
    step_sizes = np.asarray(step_sizes)
    degenerate = step_sizes < DEGENERATE_STEP_SIZE
    num_degenerate = int(degenerate.sum())

    logger.info(
        "%s adapted step sizes: median %.4g, min %.4g, max %.4g",
        label, float(np.median(step_sizes)), float(step_sizes.min()), float(step_sizes.max()),
    )
    if num_degenerate:
        datasets = sorted({int(i) for i in np.argwhere(degenerate)[:, 0]})
        logger.warning(
            "%s: %d of %d chains adapted to a step size below %.1g, in data sets %s. "
            "Those chains did not converge — their draws are kept in the file but should be "
            "dropped before pooling. This is reported rather than repaired: overwriting a "
            "collapsed step size leaves the chain under-dispersed and still breaking R-hat.",
            label, num_degenerate, step_sizes.size, DEGENERATE_STEP_SIZE, datasets,
        )
    return num_degenerate


def _build_recovery_datatree(samples, data, true_params, prior_ds, param_names, step_sizes,
                             attrs=None):
    """Build an ArviZ-compatible DataTree for all recovered datasets.

    Args:
        samples: MCMC draws, shape ``(num_datasets, num_draws, num_chains, num_params)`` —
            draws before chains, because ``inference_loop_multiple_chains`` scans over draws
            with the chain axis inside, and ``jax.vmap`` then prepends the dataset axis.
            Already in original (non-log) space.
        data: Observed data, shape ``(num_datasets, num_trials, num_cols)``.
        true_params: True parameter values, shape ``(num_datasets, num_params)``.
        prior_ds: Pre-built ``xr.Dataset`` of prior samples.
        param_names: Parameter names, length ``num_params``.
        step_sizes: Adapted step size per data set per chain, shape
            ``(num_datasets, num_chains)``. Stored as a ``sample_stats`` variable rather than
            an attribute — netCDF attributes are one-dimensional — so a collapsed adaptation
            can be located rather than merely counted.
        attrs: Extra scalar metadata merged into the tree's attributes.

    Returns:
        ``xr.DataTree`` with posterior, sample_stats, observed_data, constant_data and prior
        groups, ready
        to save as netCDF. Data sets are stored under a ``subject`` coordinate because that
        is what ArviZ and the notebooks expect, though they are independent fits rather than
        subjects of one model.
    """
    num_subjects = data.shape[0]
    subjects = np.arange(num_subjects)

    # samples: (datasets, draws, chains, params) -> per param: (chains, draws, datasets)
    posterior_ds = az.dict_to_dataset(
        {name: np.transpose(np.array(samples[..., i]), (2, 1, 0))
         for i, name in enumerate(param_names)},
        dims={name: ["subject"] for name in param_names},
        coords={"subject": subjects},
    )

    col_names = list(DATA_COL_NAMES[:data.shape[-1]])
    observed_ds = xr.Dataset(
        {name: (["subject", "trial"], np.array(data[:, :, i]))
         for i, name in enumerate(col_names)},
        coords={"subject": subjects},
    )

    constant_ds = xr.Dataset(
        {"theta": (["subject", "param"], np.array(true_params))},
        coords={"subject": subjects, "param": list(param_names)},
    )

    sample_stats_ds = xr.Dataset(
        {"step_size": (["chain", "subject"], np.asarray(step_sizes).T)},
        coords={"subject": subjects, "chain": np.arange(np.shape(step_sizes)[1])},
    )

    dt = xr.DataTree.from_dict({
        "posterior":     posterior_ds,
        "sample_stats":  sample_stats_ds,
        "observed_data": observed_ds,
        "constant_data": constant_ds,
        "prior":         prior_ds,
    })
    dt.attrs["param_names"] = list(param_names)
    dt.attrs.update(attrs or {})
    return dt


@hydra.main(version_base=None, config_path="../conf_jax", config_name="config")
def main(cfg):
    configure_jax(cfg["device"], require_device=cfg["require_device"])

    model_cfg = cfg["model"]
    spec = spec_for(model_cfg["spec"])
    param_names = list(spec.names)
    context_names = list(model_cfg["context_names"])

    train_key = jax.random.key(cfg["train_seed"])
    conditioner_key, _ = jax.random.split(train_key, 2)

    conditioner = make_mlp_conditioner(
        num_in=len(context_names),
        num_bins=model_cfg["num_bins"],
        num_mid=model_cfg["num_mid"],
        **flow_options(model_cfg),
        rngs=nnx.Rngs(default=conditioner_key),
    )

    conditioner_path = Path(cfg["conditioner_dir"]).absolute() / "conditioner"
    logger.info("Loading conditioner from: %s", conditioner_path)

    # `context_names` is checked against the checkpoint's sidecar where there is one, which
    # is what turns a reordered conditioning set — same width, different meaning — from a
    # silently wrong density into an error.
    conditioner = load_conditioner(conditioner, conditioner_path, context_names=context_names)
    conditioner.eval()

    # Make sure that different numbers of trials have different base seeds
    test_key = jax.random.key(cfg["test_seed"] + cfg["test_num_obs"])
    data_key, sampling_key, prior_sample_key = jax.random.split(test_key, 3)

    prior = instantiate(model_cfg["recovery_prior"])
    test_sampler = instantiate(model_cfg["test_sampler"])

    num_datasets = cfg["test_num_datasets"]
    num_chains = cfg["mcmc"]["num_chains"]

    log_prior = log_prior_fn(prior)

    test_data, test_context = test_sampler(data_key, (num_datasets, cfg["test_num_obs"]), prior)

    prior_raw = prior.sample(seed=prior_sample_key, sample_shape=(1000,))
    prior_samples = jnp.stack(prior_raw, axis=1)  # (1000, num_params)
    prior_ds = xr.Dataset(
        {name: (["draw"], np.array(prior_samples[:, i])) for i, name in enumerate(param_names)},
        coords={"draw": np.arange(prior_samples.shape[0])},
    )

    likelihood_factory_approx = instantiate(model_cfg["likelihood_factory_approx"])(conditioner)

    def draw_from_prior(key):
        """One unconstrained (log-space) prior draw, the dispersion source for a chain."""
        return jnp.log(jnp.stack(prior.sample(seed=key)))

    def recover_dataset(sampling_key, data, create_likelihood_fun):
        """Fit one data set with NUTS and return draws on the natural scale.

        Vmapped over data sets by the caller, so everything inside must be shape-static and
        free of Python branching on traced values.

        Args:
            sampling_key: PRNG key for this data set.
            data: One data set, ``(num_trials, num_cols)``.
            create_likelihood_fun: Takes `data`, returns a per-trial log-likelihood function
                of log-parameters.

        Returns:
            ``(draws, num_exhausted, step_sizes)``. `draws` has shape
            ``(num_draws, num_chains, num_params)`` on the natural scale; nothing is
            discarded as burn-in, warm-up already ran separately. The other two are
            diagnostics the caller aggregates and logs.
        """
        likelihood_fun = create_likelihood_fun(data)

        def logdensity_fun(x):
            return log_prior(x) + jnp.sum(likelihood_fun(x))

        init_key, warmup_key, loop_key = jax.random.split(sampling_key, 3)

        # The censoring sentinel is excluded before the minimum, so a trial that never
        # crossed cannot masquerade as the fastest response and cap `t0` at a negative value.
        support = T0Support.from_spec(
            spec,
            min_valid_rt(data[:, 0]),
            max_fraction=cfg["init"]["t0_max_fraction"],
        )
        init_positions, num_exhausted = init_positions_from_prior(
            draw_from_prior, num_chains, init_key,
            support=support,
            max_attempts=cfg["init"]["t0_rejection_max_attempts"],
        )

        # One window adaptation per chain, so the starts are dispersed *and* converged and
        # R-hat measures between-chain disagreement rather than Monte-Carlo noise.
        last_states, kernel_params = window_adaptation(
            blackjax.nuts,
            logdensity_fun,
            init_positions,
            cfg["mcmc"]["num_warmup"],
            warmup_key,
            num_chains=num_chains,
        )

        # Each chain carries its own step size and mass matrix, so the kernel takes them per
        # call instead of having them bound up front. The tuning is used as it comes.
        nuts_kernel = blackjax.nuts.build_kernel()

        def kernel(key, state, params):
            return nuts_kernel(key, state, logdensity_fun, **params)

        positions, _ = inference_loop_multiple_chains(
            loop_key, kernel, last_states, cfg["mcmc"]["num_sampling"], num_chains, kernel_params,
        )

        return spec.constrain(positions), num_exhausted, kernel_params["step_size"]

    def run(label, key, create_likelihood_fun, path):
        """Fit every data set, report the diagnostics, and write the netCDF."""
        keys = jax.random.split(key, num_datasets)
        samples, exhausted, step_sizes = jax.vmap(recover_dataset, in_axes=(0, 0, None))(
            keys, test_data, create_likelihood_fun
        )
        samples.block_until_ready()

        logger.info("%s samples shape: %s", label, samples.shape)
        _log_init_exhaustion(label, exhausted, num_datasets, num_chains)
        num_degenerate = _log_step_sizes(label, step_sizes)

        dt = _build_recovery_datatree(
            samples=np.array(samples),
            data=np.array(test_data),
            true_params=np.array(test_context),
            prior_ds=prior_ds,
            param_names=param_names,
            step_sizes=step_sizes,
            attrs={
                "num_degenerate_chains": num_degenerate,
                "num_init_exhausted": int(jnp.sum(exhausted)),
            },
        )
        logger.info("Saving %s results to %s", label, path)
        dt.to_netcdf(path)

    sampling_key_approx, sampling_key_ref = jax.random.split(sampling_key, 2)

    run("approx", sampling_key_approx, likelihood_factory_approx, "parameter_recovery_approx.nc")

    if model_cfg.get("run_reference_recovery", False):
        run(
            "ref",
            sampling_key_ref,
            instantiate(model_cfg["likelihood_factory_ref"]),
            "parameter_recovery_ref.nc",
        )


if __name__ == "__main__":
    main()
