
import logging
from functools import partial
from pathlib import Path

import blackjax
import blackjax.smc.resampling as resampling
import hydra
from hydra.utils import instantiate
import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import numpy as np
from blackjax.smc import extend_params
from flax import nnx
from omegaconf import OmegaConf
from tensorflow_probability.substrates.jax import bijectors as tfb
from confrdm.data import save_hdf5
from confrdm_jax.flows import load_conditioner
from confrdm_jax.flows import make_mlp_conditioner
from confrdm_jax.mcmc import warmup
from confrdm_jax.smc import smc_inference_loop

logger = logging.getLogger(__name__)

logging.getLogger("absl").setLevel(logging.ERROR)

jax.config.update('jax_enable_x64', True)


def sample_prior_particles(prior, bijector, num_particles, rng_key):
    """Draw particles from the hierarchical prior in unconstrained flat space.

    Samples from the prior, maps to unconstrained space via bijector.inverse,
    and ravels each sample to a flat vector.

    Args:
        prior: Hierarchical prior instance.
        bijector: TFP JointMap bijector (constrained <-> unconstrained).
        num_particles: Number of particles to draw.
        rng_key: JAX PRNG key.

    Returns:
        Array of shape (num_particles, num_flat_params).
    """
    keys = jax.random.split(rng_key, num_particles)

    def sample_and_ravel(key):
        sample = prior.sample(seed=key)
        unconstrained = bijector.inverse(sample)
        flat, _ = jfu.ravel_pytree(unconstrained)
        return flat

    return jax.vmap(sample_and_ravel)(keys)


@hydra.main(version_base=None, config_path="../conf_jax", config_name="config")
def main(cfg):
    hier_cfg = cfg["hierarchical_recovery"]
    model_cfg = cfg["model"]
    model_hier_cfg = model_cfg["hierarchical"]

    num_subjects = hier_cfg["num_subjects"]
    num_trials = hier_cfg["test_num_obs_per_subject"]
    num_populations = hier_cfg["test_num_populations"]

    num_params = model_hier_cfg["num_params"]

    # Derived constants
    num_L_params = num_params + num_params * (num_params - 1) // 2
    num_pop_params = num_params + num_L_params

    # Prior hyperparameters from model-specific config
    prior_cfg = OmegaConf.to_container(model_hier_cfg["prior"], resolve=True)

    # Instantiate model-specific functions from config _target_ entries
    prior_factory = instantiate(model_hier_cfg["prior_factory"])
    sampler = instantiate(model_hier_cfg["test_sampler"])
    likelihood_factory_fn = instantiate(model_hier_cfg["likelihood_factory_approx"])

    # Build CholeskyLKJ-MVN prior distribution
    prior = prior_factory(num_subjects, **prior_cfg)

    # Load conditioner
    train_key = jax.random.key(cfg["train_seed"])
    conditioner_key, _ = jax.random.split(train_key, 2)

    rngs = nnx.Rngs(default=conditioner_key)
    conditioner = make_mlp_conditioner(
        num_in=model_cfg["num_params"],
        num_bins=model_cfg["num_bins"],
        num_mid=model_cfg["num_mid"],
        rngs=rngs,
    )

    conditioner_path = Path(cfg["conditioner_dir"]).absolute() / "conditioner"
    logger.info("Loading conditioner from: %s", conditioner_path)
    conditioner = load_conditioner(conditioner, conditioner_path)
    conditioner.eval()

    # Create likelihood factory
    likelihood_factory_approx = likelihood_factory_fn(conditioner)

    smc_cfg = hier_cfg["smc"]

    bijector = tfb.JointMap({
        "s": tfb.Exp(),                 # HalfNormal -> Positive
        "mu": tfb.Identity(),           # Normal -> Real
        "psi_raw": tfb.CorrelationCholesky(), # Matrix -> Cholesky Factor
        "z": tfb.Identity()             # Normal -> Real
    })

    def log_prior_fn(flat_params):
        unconstrained_params_dict = unravel_fn(flat_params)
        params = bijector.forward(unconstrained_params_dict)
        prior_lp = prior.log_prob(params)
        ljc = bijector.forward_log_det_jacobian(unconstrained_params_dict)
        return prior_lp + jnp.sum(ljc)

    def recover_population(sampling_key, data, mask, create_likelihood_fun, init_position):
        """Run tempered SMC to recover hierarchical parameters for one population."""
        likelihood_fun = create_likelihood_fun(data, mask)

        # Wrap the centered likelihood to accept non-centered parameterization
        def log_likelihood_fn_wrapped(flat_params):
            unconstrained_params_dict = unravel_fn(flat_params)
            params = bijector.forward(unconstrained_params_dict)
            # Non-centered reconstruction logic
            psi = params['s'][:, None] * params['psi_raw']
            theta = params['mu'] + jnp.einsum('nj,ij->ni', params['z'], psi)
            return jnp.sum(likelihood_fun(theta))

        def logdensity_fn(params):
            return log_prior_fn(params) + log_likelihood_fn_wrapped(params)

        # Run window adaptation to find good HMC parameters
        sampling_key, warmup_key = jax.random.split(sampling_key)
        _, _, adapted_params = warmup(
            blackjax.nuts,
            logdensity_fn,  # Requires logdensity = logprior + loglikelihood
            init_position,
            smc_cfg["num_warmup"],
            warmup_key,
        )
        logger.info(
            "Adapted step size: %s", adapted_params["step_size"],
        )

        hmc_parameters = dict(
            step_size=adapted_params["step_size"],
            inverse_mass_matrix=adapted_params["inverse_mass_matrix"],
            num_integration_steps=smc_cfg["num_integration_steps"],
        )

        tempered = blackjax.adaptive_tempered_smc(
            log_prior_fn,
            log_likelihood_fn_wrapped, # Requires only loglikelihood
            blackjax.hmc.build_kernel(),
            blackjax.hmc.init,
            extend_params(hmc_parameters),
            resampling.systematic,
            smc_cfg["target_ess"],
            num_mcmc_steps=smc_cfg["num_mcmc_steps"],
        )

        # Sample initial particles from the prior
        sampling_key, particle_key = jax.random.split(sampling_key)
        initial_particles = sample_prior_particles(
            prior, bijector, smc_cfg["num_particles"], particle_key,
        )

        initial_state = tempered.init(initial_particles)

        # Run SMC with multiple chains
        num_chains = smc_cfg["num_chains"]
        sample_keys = jax.random.split(sampling_key, num_chains)

        n_iter, final_state = jax.vmap(
            smc_inference_loop, in_axes=(0, None, None),
        )(sample_keys, tempered.step, initial_state)

        return final_state, n_iter

    # Generate and recover for each population
    test_key = jax.random.key(cfg["test_seed"])
    all_results = {}

    for pop_idx in range(num_populations):
        logger.info("Population %d / %d", pop_idx + 1, num_populations)

        test_key, data_key, sampling_key = jax.random.split(test_key, 3)

        # Simulate hierarchical data using LKJ-MVN prior
        data, context = sampler(
            data_key, num_trials, num_subjects, **prior_cfg,
        )
        logger.info("Simulated data shape: %s", data.shape)

        # All subjects have the same trial count, mask is all True
        mask = jnp.ones((num_subjects, num_trials), dtype=bool)

        # Build initial position from true params (used for warmup adaptation)
        init_position, unravel_fn = jfu.ravel_pytree(bijector.inverse(context))

        # Approximate recovery
        logger.info("Running approximate recovery...")
        sampling_key, approx_key = jax.random.split(sampling_key)

        final_state, n_iter = recover_population(
            approx_key, data, mask, likelihood_factory_approx, init_position,
        )
        logger.info("SMC converged in %s iterations", n_iter)

        # Final particles shape: (num_chains, num_particles, num_params)
        pop_results = {
            f"pop{pop_idx}_particles_approx": np.asarray(final_state.particles),
            f"pop{pop_idx}_log_weights_approx": np.asarray(final_state.weights),
            f"pop{pop_idx}_data": np.asarray(data),
        }

        for key, val in context.items():
            pop_results[f"pop{pop_idx}_true_{key}"] = val

        # Reference recovery (optional)
        if cfg["model"].get("run_reference_recovery", False):
            logger.info("Running reference recovery...")
            sampling_key, ref_key = jax.random.split(sampling_key)

            ref_likelihood = instantiate(model_hier_cfg["likelihood_factory_ref"])
            final_state_ref, n_iter_ref = recover_population(
                ref_key, data, mask, ref_likelihood, init_position,
            )
            logger.info("Ref SMC converged in %s iterations", n_iter_ref)
            pop_results[f"pop{pop_idx}_particles_ref"] = np.asarray(final_state_ref.particles)
            pop_results[f"pop{pop_idx}_log_weights_ref"] = np.asarray(final_state_ref.weights)

        all_results.update(pop_results)

    logger.info("Saving results")
    save_hdf5("parameter_recovery_hierarchical.hdf5", all_results)


if __name__ == "__main__":
    main()
