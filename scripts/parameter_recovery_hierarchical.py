
import logging
from pathlib import Path

import blackjax
import hydra
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from tensorflow_probability.substrates.jax import distributions as tfd

from confrdm.data import save_hdf5
from confrdm_jax.flows import load_conditioner
from confrdm_jax.flows import make_mlp_conditioner
from confrdm_jax.likelihoods import create_rdm_hierarchical_likelihood
from confrdm_jax.likelihoods import create_rdm_hierarchical_likelihood_factory_approx
from confrdm_jax.mcmc import inference_loop_multiple_chains
from confrdm_jax.mcmc import warmup
from confrdm_jax.simulators import create_rdm_hierarchical_prior
from confrdm_jax.simulators import sample_conditional_rdm_hierarchical

logger = logging.getLogger(__name__)

logging.getLogger("absl").setLevel(logging.ERROR)

NUM_RDM_PARAMS = 5

# The hierarchical prior sample has 5 parameter groups:
#   v_intercept: (mu, sigma, subjects)    - truncated normal, 2 pop scalars
#   v_slope:     (mu, sigma, subjects)    - truncated normal, 2 pop scalars
#   s_true:      (scale, subjects)        - gamma, 1 pop scalar
#   b:           (scale, subjects)        - gamma, 1 pop scalar
#   t0:          (mu, sigma, subjects)    - truncated normal, 2 pop scalars
# Total pop scalars: 8
#
# The flat MCMC vector layout (no padding):
#   [pop_0, pop_1, ..., pop_7, subj_0_p0, subj_0_p1, ..., subj_S_pP]

GROUP_POP_SIZES = [2, 2, 1, 1, 2]  # actual pop scalars per group
NUM_POP_PARAMS = sum(GROUP_POP_SIZES)  # 8

# Hyperprior config matching create_rdm_hierarchical_prior.
# Each entry: ("truncated_normal", (hyper_mu_mu, hyper_mu_s, hyper_s))
#          or ("gamma", (hyper_s_mu, hyper_s_s, gamma_shape))
HYPERPRIOR_CONFIG = [
    ("truncated_normal", (1.0, 0.25, 0.5)),   # v_intercept
    ("truncated_normal", (2.5, 0.25, 0.5)),    # v_slope
    ("gamma", (0.1, 0.05, 12.0)),              # s_true
    ("gamma", (0.15, 0.05, 8.0)),              # b
    ("truncated_normal", (0.3, 0.2, 0.1)),     # t0
]


def truncnorm_logpdf(x, loc, scale):
    """Log-density of TruncatedNormal(loc, scale, low=0, high=inf).

    TFP's TruncatedNormal.log_prob has broken gradients w.r.t. scale,
    so we implement it manually using jax.scipy.stats.norm.
    """
    z = (x - loc) / scale
    log_pdf = jax.scipy.stats.norm.logpdf(z) - jnp.log(scale)
    log_normalizer = jax.scipy.stats.norm.logcdf(loc / scale)
    return log_pdf - log_normalizer


def context_to_flat(context):
    """Convert hierarchical prior sample to flat vector for MCMC.

    Returns a vector of length NUM_POP_PARAMS + S*P in natural (not log) space.
    Pop scalars are packed without padding (8 values for RDM).
    """
    pop_scalars = []
    subj_columns = []

    for group in context:
        for scalar in group[:-1]:
            pop_scalars.append(scalar)
        subj_columns.append(group[-1])

    pop = jnp.array(pop_scalars)  # length 8
    subj = jnp.stack(subj_columns, axis=-1)  # (S, P)

    return jnp.concatenate([pop, subj.ravel()])


def flat_to_prior_sample(x, num_subjects):
    """Convert flat parameter vector back to nested prior sample structure.

    Inverse of context_to_flat.
    """
    pop = x[:NUM_POP_PARAMS]
    subj = x[NUM_POP_PARAMS:].reshape(num_subjects, NUM_RDM_PARAMS)

    sample = []
    pop_idx = 0
    for param_idx, num_pop in enumerate(GROUP_POP_SIZES):
        group = [pop[pop_idx + i] for i in range(num_pop)]
        group.append(subj[:, param_idx])
        sample.append(group)
        pop_idx += num_pop

    return sample


def context_to_true_params(context):
    """Extract true parameters from context for saving.

    Returns:
        pop_params: 1-D array of all population-level scalars (8 values).
        subj_params: (S, P) array of subject-level parameters.
    """
    pop_scalars = []
    subj_columns = []

    for group in context:
        for scalar in group[:-1]:
            pop_scalars.append(float(scalar))
        subj_columns.append(group[-1])

    subj = jnp.stack(subj_columns, axis=-1)
    return np.array(pop_scalars), np.asarray(subj)


@hydra.main(version_base=None, config_path="../conf_jax", config_name="config")
def main(cfg):
    num_subjects = cfg["hierarchical_recovery"]["num_subjects"]
    num_trials = cfg["hierarchical_recovery"]["test_num_obs_per_subject"]
    num_populations = cfg["hierarchical_recovery"]["test_num_populations"]

    # Load conditioner
    train_key = jax.random.key(cfg["train_seed"])
    conditioner_key, _ = jax.random.split(train_key, 2)

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

    # Create hierarchical prior and likelihood factory
    prior = create_rdm_hierarchical_prior(num_subjects)
    likelihood_factory_approx = create_rdm_hierarchical_likelihood_factory_approx(conditioner)

    mcmc_cfg = cfg["hierarchical_recovery"]["mcmc"]

    def log_prior(x):
        """Log-prior on the flat MCMC vector (log-space).

        Evaluates each hyperprior group manually using truncnorm_logpdf to avoid
        NaN gradients from TFP's TruncatedNormal.log_prob w.r.t. scale.
        Includes the Jacobian correction for the exp transform.
        """
        x_natural = jnp.exp(x)
        sample = flat_to_prior_sample(x_natural, num_subjects)

        lp = jnp.float32(0.0)
        for group, (group_type, hyperparams) in zip(sample, HYPERPRIOR_CONFIG):
            if group_type == "truncated_normal":
                mu, sigma, subjects = group
                hyper_mu_mu, hyper_mu_s, hyper_s = hyperparams
                lp += truncnorm_logpdf(mu, hyper_mu_mu, hyper_mu_s)
                lp += tfd.HalfNormal(hyper_s).log_prob(sigma)
                lp += jnp.sum(truncnorm_logpdf(subjects, mu, sigma))
            else:
                scale, subjects = group
                hyper_s_mu, hyper_s_s, gamma_shape = hyperparams
                lp += truncnorm_logpdf(scale, hyper_s_mu, hyper_s_s)
                lp += jnp.sum(tfd.Gamma(gamma_shape, 1.0 / scale).log_prob(subjects))

        # Jacobian correction for log-space parameterization: |d(exp(x))/dx| = exp(x)
        # lp += jnp.sum(x)

        return lp

    def recover_population(sampling_key, data, mask, create_likelihood_fun, init_params):
        """Run MCMC to recover hierarchical parameters for one population."""
        likelihood_fun = create_likelihood_fun(data, mask)

        def logdensity_fun(x):
            return log_prior(x) + likelihood_fun(x)

        sampling_key, warmup_key = jax.random.split(sampling_key)

        kernel, last_state, _ = warmup(
            blackjax.nuts,
            logdensity_fun,
            init_params,
            mcmc_cfg["num_warmup"],
            warmup_key,
        )

        num_chains = mcmc_cfg["num_chains"]
        last_states = jax.vmap(lambda _: last_state)(jnp.arange(num_chains))

        trace = inference_loop_multiple_chains(
            sampling_key, kernel, last_states, mcmc_cfg["num_sampling"], num_chains,
        )

        return jnp.exp(trace.position)

    # Generate and recover for each population
    test_key = jax.random.key(cfg["test_seed"])
    all_results = {}

    for pop_idx in range(num_populations):
        logger.info("Population %d / %d", pop_idx + 1, num_populations)

        test_key, data_key, sampling_key = jax.random.split(test_key, 3)

        # Simulate hierarchical data
        data, context = sample_conditional_rdm_hierarchical(data_key, num_trials, prior)
        logger.info("Simulated data shape: %s", data.shape)

        # All subjects have the same trial count, mask is all True
        mask = jnp.ones((num_subjects, num_trials), dtype=bool)

        # Build initial position from true params (in log-space)
        true_flat = context_to_flat(context)
        init_position = jnp.log(jnp.maximum(true_flat, 1e-6))

        # Save true parameters
        pop_params_true, subj_params_true = context_to_true_params(context)

        # Approximate recovery
        logger.info("Running approximate recovery...")
        sampling_key, approx_key = jax.random.split(sampling_key)

        samples_approx = recover_population(
            approx_key, data, mask, likelihood_factory_approx, init_position,
        )
        samples_approx.block_until_ready()
        logger.info("Approx samples shape: %s", samples_approx.shape)

        pop_results = {
            f"pop{pop_idx}_samples_approx": np.asarray(samples_approx),
            f"pop{pop_idx}_data": np.asarray(data),
            f"pop{pop_idx}_pop_params_true": pop_params_true,
            f"pop{pop_idx}_subj_params_true": np.asarray(subj_params_true),
        }

        # Reference recovery (optional)
        if cfg["model"].get("run_reference_recovery", False):
            logger.info("Running reference recovery...")
            sampling_key, ref_key = jax.random.split(sampling_key)

            samples_ref = recover_population(
                ref_key, data, mask, create_rdm_hierarchical_likelihood, init_position,
            )
            samples_ref.block_until_ready()
            logger.info("Ref samples shape: %s", samples_ref.shape)
            pop_results[f"pop{pop_idx}_samples_ref"] = np.asarray(samples_ref)

        all_results.update(pop_results)

    logger.info("Saving results")
    save_hdf5("parameter_recovery_hierarchical.hdf5", all_results)


if __name__ == "__main__":
    main()
