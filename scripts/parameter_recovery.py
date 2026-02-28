
import logging
from pathlib import Path
import blackjax
import hydra
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from hydra.utils import instantiate
from confrdm.data import save_hdf5
from confrdm_jax.flows import make_mlp_conditioner
from confrdm_jax.flows import load_conditioner
from confrdm_jax.mcmc import inference_loop_multiple_chains
from confrdm_jax.mcmc import warmup

logger = logging.getLogger(__name__)

logging.getLogger("absl").setLevel(logging.ERROR)


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
    data_key, sampling_key = jax.random.split(test_key, 2)

    # Instantiate prior and sampler from config
    prior = instantiate(cfg["model"]["recovery_prior"])
    test_sampler = instantiate(cfg["model"]["test_sampler"])

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

    # Create likelihood factory from config
    likelihood_factory_approx = instantiate(cfg["model"]["likelihood_factory_approx"])(conditioner)

    def recover_dataset(sampling_key, data, create_likelihood_fun):
        init_position = jnp.array(prior.mode())
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

    results = dict(
        samples_approx=np.asarray(samples_approx),
        data=np.asarray(test_data.squeeze()),
        true_params=np.asarray(test_context.squeeze())
    )

    # Reference recovery (optional, controlled by config)
    if cfg["model"].get("run_reference_recovery", False):
        likelihood_factory_ref = instantiate(cfg["model"]["likelihood_factory_ref"])
        sampling_keys_ref = jax.random.split(sampling_key_ref, test_data.shape[0])

        samples_ref = jax.vmap(recover_dataset, in_axes=(0, 0, None))(
            sampling_keys_ref, test_data, likelihood_factory_ref
        )
        samples_ref.block_until_ready()

        logger.info("Ref samples shape: %s", samples_ref.shape)
        results["samples_ref"] = np.asarray(samples_ref)

    logger.info("Saving results")
    save_hdf5("parameter_recovery.hdf5", results)


if __name__ == "__main__":
    main()
