
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
from confrdm_jax.mcmc import warmup

logger = logging.getLogger(__name__)

logging.getLogger("absl").setLevel(logging.ERROR)

jax.config.update('jax_enable_x64', True)

# Column names for observed data; RDM has 2 cols, CRDM has 3.
_DATA_COL_NAMES = ["rt", "choice", "condition"]


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

    test_key = jax.random.key(cfg["test_seed"])
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

    test_data = test_data.squeeze()
    test_context = test_context.squeeze()  # (num_datasets, num_params)

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
        init_position = jnp.array(prior.mode(), dtype=jnp.float64)
        init_position = init_position.at[-1].set(data[:, 0].min() / 2)

        likelihood_fun = create_likelihood_fun(data)

        def logdensity_fun(x):
            return log_prior(x) + jnp.sum(likelihood_fun(x))

        sampling_key, warmup_key = jax.random.split(sampling_key)

        kernel, last_state, _ = warmup(
            blackjax.nuts,
            logdensity_fun,
            jnp.log(init_position),
            cfg["mcmc"]["num_warmup"],
            warmup_key,
        )

        num_chains = cfg["mcmc"]["num_chains"]

        last_states = jax.vmap(lambda _: last_state)(jnp.arange(num_chains))

        positions, _ = inference_loop_multiple_chains(
            sampling_key, kernel, last_states, cfg["mcmc"]["num_sampling"], num_chains,
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
