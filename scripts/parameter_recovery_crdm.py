
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
from confrdm_jax.likelihoods import inv_gauss_log_pdf_sf
from confrdm_jax.mcmc import inference_loop_multiple_chains
from confrdm_jax.mcmc import warmup
from confrdm_jax.priors import crdm_prior
from confrdm_jax.simulators import sample_conditional_crdm_condition, create_crdm_prior_informed

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

    conditioner = load_conditioner(conditioner, Path("conditioner").absolute())
    conditioner.eval()

    test_key = jax.random.key(cfg["test_seed"])
    data_key, sampling_key = jax.random.split(test_key, 2)

    prior = create_crdm_prior_informed()

    test_data, test_context = sample_conditional_crdm_condition(
        data_key,
        (cfg["test_num_datasets"], cfg["test_num_obs"]),
        prior,
        cfg["model"]["test_dt"],
        cfg["model"]["t_max"],
    )

    test_data = test_data.squeeze()

    def crdm_log_pdf_sf(rt, v_c, amp, tau, s, b, t0):
        rt = rt - t0
        rt = jnp.maximum(0.0, rt)
        # Ensure conditioner is accessible here
        context = jnp.transpose(jnp.array([v_c, jnp.abs(amp), tau, s, b]))
        return evaluate_pdf_sf(conditioner, rt, context)

    def create_crdm_two_accumulators_likelihood_conditions(data):
        rt = data[:, 0]
        choice = data[:, 1]
        condition = data[:, 2]  # 1 = Congruent, 0 = Incongruent

        def likelihood_fun(x, min_ll=1e-12):
            # --- 1. Define Parameters ---
            v_c_true = x[0] + x[1]
            v_c_false = x[0]
            amp = x[2]
            tau = x[3]
            s_true = x[4]
            s_false = 1.0
            b = x[5]
            t0 = x[6]

            # --- 2. Input Routing (The Optimization) ---
            # We determine which parameters belong to the CRDM (NN) and which to InvGauss
            # purely based on the condition column.
            
            # If Congruent: CRDM gets (v_c_true, s_true). 
            # If Incongruent: CRDM gets (v_c_false, s_false).
            nn_v_c = jnp.where(condition == 1, v_c_true, v_c_false)
            nn_s   = jnp.where(condition == 1, s_true, s_false)

            # Conversely for InvGauss:
            # If Congruent: InvGauss gets (v_c_false, s_false).
            # If Incongruent: InvGauss gets (v_c_true, s_true).
            ig_v_c = jnp.where(condition == 1, v_c_false, v_c_true)
            ig_s   = jnp.where(condition == 1, s_false, s_true)

            amp = jnp.tile(amp, nn_v_c.shape)
            tau = jnp.tile(tau, nn_v_c.shape)
            b = jnp.tile(b, nn_v_c.shape)

            # --- 3. Efficient Execution (1x Cost) ---
            # Run Neural Network exactly ONCE per trial with the correctly routed params
            # Note: amp, tau, b, t0 seem consistent for the CRDM accumulator across conditions
            nn_log_pdf, nn_log_sf = crdm_log_pdf_sf(rt, nn_v_c, amp, tau, nn_s, b, t0)
            
            # Run Analytic Function exactly ONCE per trial
            ig_log_pdf, ig_log_sf = inv_gauss_log_pdf_sf(rt, ig_v_c, ig_s, b, t0)

            # --- 4. Result Routing ---
            # Now we map the outputs back to "Target" (True) or "Non-Target" (False) accumulators
            
            # Target Accumulator logic:
            # If Congruent: Target was CRDM (nn_result)
            # If Incongruent: Target was InvGauss (ig_result)
            log_pdf_true = jnp.where(condition == 1, nn_log_pdf.squeeze(), ig_log_pdf.squeeze())
            
            # Non-Target Accumulator logic:
            # If Congruent: Non-Target was InvGauss (ig_result)
            # If Incongruent: Non-Target was CRDM (nn_result)
            log_sf_false = jnp.where(condition == 1, ig_log_sf.squeeze(), nn_log_sf.squeeze())
            
            # ... Do the inverse for the other cross-combinations needed for the choice rule ...
            log_pdf_false = jnp.where(condition == 1, ig_log_pdf.squeeze(), nn_log_pdf.squeeze())
            log_sf_true   = jnp.where(condition == 1, nn_log_sf.squeeze(), ig_log_sf.squeeze())

            # --- 5. Final Choice Logic ---
            dens_choice_1 = log_pdf_true + log_sf_false
            dens_choice_0 = log_pdf_false + log_sf_true

            ll = jnp.where(choice == 1, dens_choice_1, dens_choice_0)

            return jnp.where(jnp.isfinite(ll), ll, jnp.log(min_ll))

        return likelihood_fun

    def recover_dataset(sampling_key, data):
        init_position = jnp.array(prior.mode()) 
        init_position = init_position.at[-1].set(data[:, 0].min() / 2)

        likelihood_fun = create_crdm_two_accumulators_likelihood_conditions(data)

        def logdensity_fun(x):
            return crdm_prior(x) + jnp.sum(likelihood_fun(jnp.exp(x)))

        sampling_key, warmup_key = jax.random.split(sampling_key)

        # Warmup
        kernel, last_state, _ = warmup(
            blackjax.nuts,
            logdensity_fun,
            jnp.log(init_position),
            cfg["mcmc"]["num_warmup"],
            warmup_key,
        )

        num_chains = cfg["mcmc"]["num_chains"]

        # Prepare states for multiple chains
        last_states = jax.vmap(lambda _: last_state)(jnp.arange(num_chains))

        # Sampling
        trace = inference_loop_multiple_chains(
            sampling_key, kernel, last_states, cfg["mcmc"]["num_sampling"], num_chains,
        )

        return jnp.exp(trace.position)

    sampling_keys = jax.random.split(sampling_key, test_data.shape[0])

    samples_approx = jax.vmap(recover_dataset, in_axes=(0, 0))(sampling_keys, test_data)
    samples_approx.block_until_ready()

    logger.info(samples_approx.shape)

    results = dict(
        samples_approx=np.asarray(samples_approx),
        data=np.asarray(test_data.squeeze()),
        true_params=np.asarray(test_context.squeeze())
    )
    
    logger.info("Saving results")

    save_hdf5("parameter_recovery.hdf5", results)


if __name__ == "__main__":
    main()
