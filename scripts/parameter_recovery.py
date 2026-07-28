
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


def _overdispersed_init(rng_key, prior, num_chains, min_rt):
    """Draw one starting position per chain, dispersed but inside the support.

    Prior draws supply the dispersion for every parameter except `t0`, which
    cannot be treated the same way: the likelihood is undefined for
    `t0 >= min(rt)` and the penalty branch there has a gradient of order 1e3,
    so a chain started above the boundary diverges on every step and never
    recovers. Measured: perturbing all five parameters alike gives 1001/1000
    divergences and ESS 7 (CONFRDM_JAX.md §8.4).

    `t0` is therefore spread deterministically across the interval it is
    actually confined to. It is the last parameter by the ordering contract in
    CONFRDM_JAX.md §2.
    """
    draws = jnp.stack(prior.sample(seed=rng_key, sample_shape=(num_chains,)), axis=1)
    t0_fraction = jnp.linspace(0.2, 0.8, num_chains)
    return jnp.log(draws.at[:, -1].set(t0_fraction * min_rt))


def _build_recovery_datatree(samples, data, true_params, prior_ds, param_names):
    """Build an ArviZ-compatible DataTree for all recovered datasets.

    Args:
        samples: MCMC draws, shape ``(num_subjects, num_chains, num_draws, num_params)``,
            already in original (non-log) space.
        data: Observed data, shape ``(num_subjects, num_trials, num_cols)``.
        true_params: True parameter values, shape ``(num_subjects, num_params)``.
        prior_ds: Pre-built ``xr.Dataset`` of prior samples.
        param_names: List of parameter names, length ``num_params``.

    Returns:
        ``xr.DataTree`` with posterior, observed_data, constant_data, and
        prior groups, ready to save as netCDF.
    """
    num_subjects = data.shape[0]
    subjects = np.arange(num_subjects)

    # samples: (subjects, draws, chains, params) → per param: (chains, draws, subjects)
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

    param_names = cfg["model"]["hierarchical"]["param_names"]

    @jax.jit
    def log_prior(x):
        params = jnp.exp(x)
        return prior.log_prob([params[i] for i in range(params.shape[0])]) + jnp.sum(x) # Add the Jacobian of the log-transform

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

        init_positions = _overdispersed_init(init_key, prior, num_chains, min_rt)

        # One window adaptation per chain, so the starts are dispersed *and*
        # converged and R-hat measures between-chain disagreement rather than
        # Monte-Carlo noise.  See CONFRDM_JAX.md §8.
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

        return jnp.exp(positions)

    # Approximate recovery
    sampling_key_approx, sampling_key_ref = jax.random.split(sampling_key, 2)
    sampling_keys_approx = jax.random.split(sampling_key_approx, test_data.shape[0])

    samples_approx = jax.vmap(recover_dataset, in_axes=(0, 0, None))(
        sampling_keys_approx, test_data, likelihood_factory_approx
    )
    samples_approx.block_until_ready()

    logger.info("Approx samples shape: %s", samples_approx.shape)

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

        samples_ref = jax.vmap(recover_dataset, in_axes=(0, 0, None))(
            sampling_keys_ref, test_data, likelihood_factory_ref
        )
        samples_ref.block_until_ready()
        logger.info("Ref samples shape: %s", samples_ref.shape)

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
