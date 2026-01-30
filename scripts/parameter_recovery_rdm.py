
import logging
from pathlib import Path
import blackjax
import hydra
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from confrdm.data import save_hdf5
from confrdm_jax.flows import evaluate_pdf_sf
from confrdm_jax.flows import make_mlp_conditioner
from confrdm_jax.flows import load_conditioner
from confrdm_jax.likelihoods import create_rdm_two_accumulators_likelihood
from confrdm_jax.mcmc import inference_loop_multiple_chains
from confrdm_jax.mcmc import warmup
from confrdm_jax.priors import rdm_prior
from confrdm_jax.simulators import sample_conditional_rdm, create_rdm_prior_informed

logger = logging.getLogger(__name__)

logging.getLogger("absl").setLevel(logging.ERROR)


@hydra.main(version_base=None, config_path="../conf_jax", config_name="config")
def main(cfg):

    train_key = jax.random.key(cfg["train_seed"])
    conditioner_key, sampling_key = jax.random.split(train_key, 2)

    rngs = nnx.Rngs(default=conditioner_key, sampling=sampling_key)

    conditioner = make_mlp_conditioner(
        num_in=cfg["model"]["num_params"],
        num_bins=cfg["model"]["num_bins"],
        num_mid=cfg["model"]["num_mid"],
        rngs=rngs,
    )

    conditioner = load_conditioner(conditioner, Path("conditioner").absolute())
    conditioner.eval()

    test_key = jax.random.key(cfg["test_seed"])
    data_key, sampling_key = jax.random.split(test_key, 2)

    prior = create_rdm_prior_informed()

    test_data, test_context = sample_conditional_rdm(data_key, (cfg["test_num_datasets"], cfg["test_num_obs"]), prior)

    def inv_gauss_log_pdf_sf(rt, v, b, s, t0):
        rt = rt - t0
        rt = jnp.maximum(0.0, rt)

        return evaluate_pdf_sf(conditioner, rt, jnp.array([v, s, b]))

    def create_rdm_two_accumulators_likelihood_approx(data):
        rt = data[:, 0]
        choice = data[:, 1]

        @nnx.jit
        def likelihood_fun(x, min_ll=1e-12):
            x = jnp.exp(x)

            v_true = x[0] + x[1]
            v_false = x[0]
            s_true = x[2]
            s_false = 1.0
            b = x[3]
            t0 = x[4]

            log_pdf_true, log_sf_true = inv_gauss_log_pdf_sf(rt, v_true, b, s_true, t0)
            log_pdf_false, log_sf_false = inv_gauss_log_pdf_sf(rt, v_false, b, s_false, t0)

            dens_true = log_pdf_true.squeeze() + log_sf_false.squeeze()
            dens_false = log_pdf_false.squeeze() + log_sf_true.squeeze()

            ll = jnp.where(choice == 1, dens_true, dens_false)

            return jnp.where(jnp.isfinite(ll), ll, jnp.log(min_ll))

        return likelihood_fun

    def recover_dataset(sampling_key, data, create_likelihood_fun):
        init_position = jnp.array(prior.mode())
        init_position = init_position.at[-1].set(data[:, 0].min() / 2)

        likelihood_fun = create_likelihood_fun(data)

        def logdensity_fun(x):
            return rdm_prior(x) + jnp.sum(likelihood_fun(x))

        sampling_key, warmup_key = jax.random.split(sampling_key)

        kernel, last_state, _ = warmup(
            blackjax.nuts,
            logdensity_fun,
            jnp.log(init_position),
            cfg["mcmc"]["num_warmup"],
            warmup_key,
        )

        num_chains = cfg["mcmc"]["num_chains"]

        last_states = jax.vmap(lambda x: last_state)(jnp.arange(num_chains))

        trace = inference_loop_multiple_chains(
            sampling_key, kernel, last_states, cfg["mcmc"]["num_sampling"], num_chains,
        )

        samples = jnp.moveaxis(jnp.exp(trace.position), (0, 1, 2), (2, 1, 0))

        return samples
    
    sampling_keys = jax.random.split(sampling_key, test_data.shape[0])

    samples_approx = jax.vmap(recover_dataset, in_axes=(0, 0, None))(sampling_keys, test_data, create_rdm_two_accumulators_likelihood_approx)
    samples_approx.block_until_ready()

    logger.info(samples_approx.shape)

    samples_ref = jax.vmap(recover_dataset, in_axes=(0, 0, None))(sampling_keys, test_data, create_rdm_two_accumulators_likelihood)
    samples_ref.block_until_ready()

    logger.info(samples_ref.shape)

    results = dict(
        samples=np.asarray(samples_approx),
        samples_ref=np.asarray(samples_ref),
        data=np.asarray(test_data.squeeze()),
        true_params=np.asarray(test_context.squeeze())
    )
    
    logger.info("Saving results")

    save_hdf5("parameter_recovery.hdf5", results)


if __name__ == "__main__":
    main()
