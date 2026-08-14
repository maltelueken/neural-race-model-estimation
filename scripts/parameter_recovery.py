"""Single-subject parameter recovery for a trained conditioner.

Simulates independent data sets from the recovery prior, fits each one with
NUTS, and writes the posteriors to netCDF for the notebooks to analyse.  Each
data set is fit independently — they are stored under a ``subject`` coordinate
for ArviZ's benefit, but nothing is shared between them.  For the genuinely
multi-subject model see ``parameter_recovery_hierarchical.py``.

When the model has an analytic likelihood (RDM, ``run_reference_recovery:
true``) the same data are fit a second time with it, giving a reference
posterior that ``scripts/c2st_recovery.py`` can compare the neural one
against. The CRDM has no closed form, so only the approximate fit runs.

Must be launched with the same overrides that produced the checkpoint —
nothing on disk records the conditioner's architecture::

    python scripts/parameter_recovery.py model=rdm

Outputs ``parameter_recovery_approx.nc`` and, when enabled,
``parameter_recovery_ref.nc`` in the Hydra run directory.
"""

import logging
from pathlib import Path
import arviz.preview as az
import blackjax
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from flax import nnx
from hydra.utils import instantiate
from confrdm_jax.flows import make_mlp_conditioner
from confrdm_jax.flows import load_conditioner
from confrdm_jax.mcmc import inference_loop_multiple_chains
from confrdm_jax.mcmc import warmup_multiple_chains

logger = logging.getLogger(__name__)

logging.getLogger("absl").setLevel(logging.ERROR)

jax.config.update('jax_enable_x64', True)

# Column names for observed data; RDM has 2 cols, CRDM has 3.
_DATA_COL_NAMES = ["rt", "choice", "condition"]


def _sample_init_positions_in_support(
    rng_key, prior, num_chains, min_rt, *, max_t0_fraction, max_attempts,
):
    """Draw one starting position per chain from the prior restricted to the support.

    Prior draws supply the dispersion, but `t0` cannot simply be taken as
    drawn: the likelihood is undefined for `t0 >= min(rt)` and the penalty
    branch there has a gradient of order 1e3, so a chain started above the
    boundary diverges on every step and never recovers. Measured on the RDM at
    n = 500: perturbing all five parameters alike by ~0.35 in log space gives
    1001/1000 divergences, max R-hat 1.53 and min ESS 7. Dispersion has to be
    support-aware.

    Whole draws are therefore rejected and redrawn until `t0` clears
    ``max_t0_fraction * min_rt``, which is the same rule
    `sample_prior_particles_in_support` applies in
    ``parameter_recovery_hierarchical.py`` — both read the fraction and the
    budget from the ``init`` block of ``conf_jax/config.yaml``, which is also
    where the measurements behind the values are recorded. See that other
    docstring for why rejection and not a clip. Rejecting the whole draw rather
    than resampling `t0` alone keeps the starts exactly
    prior-conditional-on-support, which matters here only for tidiness — the
    single-subject prior is independent across parameters, so nothing else
    moves — but keeps the two scripts saying the same thing.

    This replaced a deterministic fan, ``t0 = linspace(0.2, 0.8) * min_rt``.
    That was in-support by construction and dispersed, but it discarded the
    prior's `t0` for a point mass per chain and its ceiling was far too low:
    the true `t0` sits at a median 0.807 of `min_rt` on the RDM and 0.926 on
    the CRDM, so ``0.8`` put every chain below the truth for 52% (RDM) and 88%
    (CRDM) of data sets and left window adaptation to walk them all back up.

    `t0` is assumed to be the **last** parameter, which holds for both recovery
    priors: RDM is ``[v_intercept, v_slope, s_true, b, t0]`` and CRDM
    ``[v_c_intercept, v_c_slope, amp, tau, s_true, b, t0]``. That ordering is
    positional and unenforced.

    Args:
        rng_key: PRNG key for the prior draws.
        prior: The recovery prior; its draws are the dispersion source.
        num_chains: Number of starting positions to produce.
        min_rt: Smallest *valid* observed RT. Callers must exclude the
            ``-1.0`` censoring sentinel before computing it.
        max_t0_fraction: Highest fraction of `min_rt` an accepted `t0` may take;
            ``init.t0_max_fraction``.
        max_attempts: Redraws before giving up on a chain and clipping it;
            ``init.t0_rejection_max_attempts``.

    Returns:
        ``(positions, num_exhausted)``. `positions` holds log-space starting
        positions of shape ``(num_chains, num_params)``; `num_exhausted` counts
        chains that hit `max_attempts` and fell back to clipping, and should be
        0 — the caller logs it.
    """
    t0_max = max_t0_fraction * min_rt

    def draw(key):
        return jnp.stack(prior.sample(seed=key))

    def draw_one(key):
        first_key, loop_key = jax.random.split(key)

        def cond(carry):
            i, sample, _ = carry
            return (sample[-1] > t0_max) & (i < max_attempts)

        def body(carry):
            i, _, k = carry
            k, draw_key = jax.random.split(k)
            return i + 1, draw(draw_key), k

        _, sample, _ = jax.lax.while_loop(
            cond, body, (0, draw(first_key), loop_key),
        )

        # Last resort for a chain that never cleared the cap; counted so the
        # caller can tell whether it ever fires.
        exhausted = sample[-1] > t0_max
        sample = sample.at[-1].set(jnp.minimum(sample[-1], t0_max))
        return jnp.log(sample), exhausted

    positions, exhausted = jax.vmap(draw_one)(jax.random.split(rng_key, num_chains))
    return positions, jnp.sum(exhausted)


def _log_init_exhaustion(label, exhausted, num_datasets, cfg):
    """Warn if any chain's `t0` rejection budget ran out and it was clipped.

    Args:
        label: Which fit produced the counts, for the message.
        exhausted: Per-data-set counts from `_sample_init_positions_in_support`.
        num_datasets: Number of data sets fit, for the denominator.
        cfg: Hydra config, read for the chain count.
    """
    total = int(jnp.sum(exhausted))
    if total:
        logger.warning(
            "%s: t0 rejection sampling exhausted for %d of %d chain starts; "
            "those fell back to clipping",
            label, total, num_datasets * cfg["mcmc"]["num_chains"],
        )


def _build_recovery_datatree(samples, data, true_params, prior_ds, param_names):
    """Build an ArviZ-compatible DataTree for all recovered datasets.

    Args:
        samples: MCMC draws, shape
            ``(num_datasets, num_draws, num_chains, num_params)`` — draws
            before chains, because ``inference_loop_multiple_chains`` scans
            over draws with the chain axis inside, and ``jax.vmap`` then
            prepends the dataset axis. Already in original (non-log) space.
        data: Observed data, shape ``(num_datasets, num_trials, num_cols)``.
        true_params: True parameter values, shape
            ``(num_datasets, num_params)``.
        prior_ds: Pre-built ``xr.Dataset`` of prior samples.
        param_names: List of parameter names, length ``num_params``.

    Returns:
        ``xr.DataTree`` with posterior, observed_data, constant_data, and
        prior groups, ready to save as netCDF. Data sets are stored under a
        ``subject`` coordinate because that is what ArviZ and the notebooks
        expect, though they are independent fits rather than subjects of one
        model.
    """
    num_subjects = data.shape[0]
    subjects = np.arange(num_subjects)

    # samples: (datasets, draws, chains, params) → per param: (chains, draws, datasets)
    posterior_ds = az.dict_to_dataset(
        {name: np.transpose(np.array(samples[..., i]), (2, 1, 0))
         for i, name in enumerate(param_names)},
        dims={name: ["subject"] for name in param_names},
        coords={"subject": subjects},
    )

    num_data_cols = data.shape[-1]
    col_names = _DATA_COL_NAMES[:num_data_cols]
    observed_ds = xr.Dataset(
        {name: (["subject", "trial"], np.array(data[:, :, i]))
         for i, name in enumerate(col_names)},
        coords={"subject": subjects},
    )

    constant_ds = xr.Dataset(
        {"theta": (["subject", "param"], np.array(true_params))},
        coords={"subject": subjects, "param": param_names},
    )

    dt = xr.DataTree.from_dict({
        "posterior":     posterior_ds,
        "observed_data": observed_ds,
        "constant_data": constant_ds,
        "prior":         prior_ds,
    })
    dt.attrs["param_names"] = param_names
    return dt


@hydra.main(version_base=None, config_path="../conf_jax", config_name="config")
def main(cfg):
    train_key = jax.random.key(cfg["train_seed"])
    conditioner_key, sampling_key = jax.random.split(train_key, 2)

    rngs = nnx.Rngs(default=conditioner_key)

    conditioner = make_mlp_conditioner(
        num_in=cfg["model"]["num_params"],
        num_bins=cfg["model"]["num_bins"],
        num_mid=cfg["model"]["num_mid"],
        rngs=rngs,
    )

    conditioner_path = Path(cfg["conditioner_dir"]).absolute() / "conditioner"

    logger.info("Loading conditioner from: %s", conditioner_path)

    conditioner = load_conditioner(conditioner, conditioner_path)
    conditioner.eval()

    # Make sure that different numbers of trials have different base seeds
    test_key = jax.random.key(cfg["test_seed"] + cfg["test_num_obs"])
    data_key, sampling_key, prior_sample_key = jax.random.split(test_key, 3)

    # Instantiate prior and sampler from config
    prior = instantiate(cfg["model"]["recovery_prior"])
    test_sampler = instantiate(cfg["model"]["test_sampler"])

    # Read from the `hierarchical` block because that is the only place the
    # names are written down; the single-subject recovery prior has the same
    # parameters in the same order, so the list is shared rather than
    # duplicated.
    param_names = cfg["model"]["hierarchical"]["param_names"]

    # MCMC works on log parameters, so the prior — which is defined on the
    # natural scale, and is the same object that generated the data — needs the
    # log-transform Jacobian: d(exp(x))/dx = exp(x), giving + sum(x).
    @jax.jit
    def log_prior(x):
        params = jnp.exp(x)
        return prior.log_prob([params[i] for i in range(params.shape[0])]) + jnp.sum(x)

    # Generate test data
    test_data, test_context = test_sampler(
        data_key,
        (cfg["test_num_datasets"], cfg["test_num_obs"]),
        prior,
    )

    # Drop only the axes that are meant to be dropped — the trailing singleton
    # the CRDM sampler appends to `data`, and the singleton prior-batch axis in
    # `context` — and never the leading dataset axis.  A blanket `.squeeze()`
    # ate that axis whenever `test_num_datasets == 1`, so `test_data.shape[0]`
    # became the trial count and `recover_dataset` was handed a single trial.
    num_datasets = cfg["test_num_datasets"]
    test_data = test_data.reshape(num_datasets, cfg["test_num_obs"], -1)
    test_context = test_context.reshape(num_datasets, -1)  # (num_datasets, num_params)

    # Sample from the prior for the prior group
    prior_raw = prior.sample(seed=prior_sample_key, sample_shape=(1000,))
    prior_samples = jnp.stack(prior_raw, axis=1)  # (1000, num_params)
    prior_ds = xr.Dataset(
        {name: (["draw"], np.array(prior_samples[:, i])) for i, name in enumerate(param_names)},
        coords={"draw": np.arange(prior_samples.shape[0])},
    )

    # Create likelihood factory from config
    likelihood_factory_approx = instantiate(cfg["model"]["likelihood_factory_approx"])(conditioner)

    def recover_dataset(sampling_key, data, create_likelihood_fun):
        """Fit one data set with NUTS and return draws on the natural scale.

        Vmapped over data sets by the caller, so everything inside must be
        shape-static and free of Python branching on traced values.

        Args:
            sampling_key: PRNG key for this data set.
            data: One data set, ``(num_trials, num_cols)``.
            create_likelihood_fun: Takes `data`, returns a per-trial
                log-likelihood function of log-parameters.

        Returns:
            ``(draws, num_exhausted)``. `draws` has shape
            ``(num_draws, num_chains, num_params)``, exponentiated back to the
            natural scale; nothing is discarded as burn-in, warm-up already ran
            separately. `num_exhausted` is the initialisation rejection budget
            overrun, which the caller sums and logs.
        """
        # Censored trials carry the sentinel rt = -1.0.  Taking the raw minimum
        # would initialise t0 negative and `jnp.log` it to NaN, silently
        # poisoning the whole chain, so initialise from valid RTs only.
        rt = data[:, 0]
        min_rt = jnp.min(jnp.where(rt > 0.0, rt, jnp.inf))

        likelihood_fun = create_likelihood_fun(data)

        def logdensity_fun(x):
            return log_prior(x) + jnp.sum(likelihood_fun(x))

        num_chains = cfg["mcmc"]["num_chains"]

        sampling_key, warmup_key, init_key = jax.random.split(sampling_key, 3)

        init_positions, num_exhausted = _sample_init_positions_in_support(
            init_key,
            prior,
            num_chains,
            min_rt,
            max_t0_fraction=cfg["init"]["t0_max_fraction"],
            max_attempts=cfg["init"]["t0_rejection_max_attempts"],
        )

        # One window adaptation per chain, so the starts are dispersed *and*
        # converged and R-hat measures between-chain disagreement rather than
        # Monte-Carlo noise.  Sharing one warmed-up state across chains leaves
        # the between-chain variance at zero on entry; on a bimodal test target
        # that reports R-hat = 1.001 while the sampler sits entirely in one
        # mode.  See `warmup_multiple_chains`.
        last_states, kernel_params = warmup_multiple_chains(
            blackjax.nuts,
            logdensity_fun,
            init_positions,
            cfg["mcmc"]["num_warmup"],
            warmup_key,
        )

        # Each chain carries its own step size and mass matrix, so the kernel
        # takes them per call instead of having them bound up front.
        nuts_kernel = blackjax.nuts.build_kernel()

        def kernel(key, state, params):
            return nuts_kernel(key, state, logdensity_fun, **params)

        positions, _ = inference_loop_multiple_chains(
            sampling_key,
            kernel,
            last_states,
            cfg["mcmc"]["num_sampling"],
            num_chains,
            kernel_params,
        )

        return jnp.exp(positions), num_exhausted

    # Approximate recovery
    sampling_key_approx, sampling_key_ref = jax.random.split(sampling_key, 2)
    sampling_keys_approx = jax.random.split(sampling_key_approx, test_data.shape[0])

    samples_approx, exhausted_approx = jax.vmap(recover_dataset, in_axes=(0, 0, None))(
        sampling_keys_approx, test_data, likelihood_factory_approx
    )
    samples_approx.block_until_ready()

    logger.info("Approx samples shape: %s", samples_approx.shape)
    _log_init_exhaustion("approx", exhausted_approx, test_data.shape[0], cfg)

    dt = _build_recovery_datatree(
        samples=np.array(samples_approx),
        data=np.array(test_data),
        true_params=np.array(test_context),
        prior_ds=prior_ds,
        param_names=param_names,
    )
    approx_path = "parameter_recovery_approx.nc"
    logger.info("Saving approx results to %s", approx_path)
    dt.to_netcdf(approx_path)

    # Reference recovery (optional, controlled by config)
    if cfg["model"].get("run_reference_recovery", False):
        likelihood_factory_ref = instantiate(cfg["model"]["likelihood_factory_ref"])
        sampling_keys_ref = jax.random.split(sampling_key_ref, test_data.shape[0])

        samples_ref, exhausted_ref = jax.vmap(recover_dataset, in_axes=(0, 0, None))(
            sampling_keys_ref, test_data, likelihood_factory_ref
        )
        samples_ref.block_until_ready()
        logger.info("Ref samples shape: %s", samples_ref.shape)
        _log_init_exhaustion("ref", exhausted_ref, test_data.shape[0], cfg)

        dt_ref = _build_recovery_datatree(
            samples=np.array(samples_ref),
            data=np.array(test_data),
            true_params=np.array(test_context),
            prior_ds=prior_ds,
            param_names=param_names,
        )
        ref_path = "parameter_recovery_ref.nc"
        logger.info("Saving ref results to %s", ref_path)
        dt_ref.to_netcdf(ref_path)


if __name__ == "__main__":
    main()
