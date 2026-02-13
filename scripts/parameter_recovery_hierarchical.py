
import logging
from functools import partial
from pathlib import Path

import blackjax
import hydra
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from confrdm.data import save_hdf5
from confrdm_jax.distributions import (
    cholesky_to_flat,
    flat_to_cholesky,
)
from confrdm_jax.flows import load_conditioner
from confrdm_jax.flows import make_mlp_conditioner
from confrdm_jax.likelihoods import create_rdm_hierarchical_likelihood
from confrdm_jax.likelihoods import create_rdm_hierarchical_likelihood_factory_approx
from confrdm_jax.mcmc import inference_loop_multiple_chains
from confrdm_jax.mcmc import warmup
from confrdm_jax.simulators import create_hierarchical_rdm_prior_lkj_mvn
from confrdm_jax.simulators import sample_conditional_rdm_hierarchical_lkj_mvn

logger = logging.getLogger(__name__)

logging.getLogger("absl").setLevel(logging.ERROR)

# jax.config.update('jax_enable_x64', True)


def context_to_flat(context):
    """Convert LKJ-MVN hierarchical prior sample (dict) to flat vector for MCMC.

    Layout: [mu (P), log_diag_L (P), offdiag_L (P*(P-1)/2), log_theta (S*P)]

    Args:
        context: Dictionary from create_rdm_hierarchical_lkj_mvn_prior.

    Returns:
        Flat vector of length NUM_POP_PARAMS + S*P.
    """
    mu = context['mu']  # (P,)
    L = context['L']    # (P, P)
    log_theta = context['log_theta']  # (S, P)

    # Flatten Cholesky factor
    L_flat = cholesky_to_flat(L)  # [log_diag, offdiag]

    return jnp.concatenate([mu, L_flat, log_theta.ravel()])


def flat_to_prior_sample(x, num_subjects, P):
    """Convert flat parameter vector back to prior sample structure.

    Inverse of context_to_flat.

    Args:
        x: Flat vector of length NUM_POP_PARAMS + S*P.
        num_subjects: Number of subjects S.
        P: Number of per-subject parameters.

    Returns:
        Dictionary with keys: 'mu', 'L', 'Sigma', 'log_theta', 'theta'.
    """
    num_pop_params = P + P + P * (P - 1) // 2
    mu = x[:P]
    L_flat = x[P:num_pop_params]
    log_theta = x[num_pop_params:].reshape(num_subjects, P)

    L = flat_to_cholesky(L_flat, P)
    Sigma = L @ L.T
    theta = jnp.exp(log_theta)

    return {
        'mu': mu,
        'L': L,
        'Sigma': Sigma,
        'log_theta': log_theta,
        'theta': theta,
    }


def context_to_true_params(context):
    """Extract true parameters from context for saving.

    Args:
        context: Dictionary from create_rdm_hierarchical_lkj_mvn_prior.

    Returns:
        Tuple of (pop_params, subj_params):
            pop_params: Dictionary with 'mu', 'L', 'Sigma', 'rho', 's'.
            subj_params: (S, P) array of subject-level parameters in natural space.
    """
    pop_params = {
        'mu': np.asarray(context['mu']),
        'L': np.asarray(context['L']),
        'Sigma': np.asarray(context['Sigma']),
        'rho': np.asarray(context['rho']),
        's': np.asarray(context['s']),
    }
    subj_params = np.asarray(context['theta'])
    return pop_params, subj_params


@hydra.main(version_base=None, config_path="../conf_jax", config_name="config")
def main(cfg):
    hier_cfg = cfg["hierarchical_recovery"]
    num_subjects = hier_cfg["num_subjects"]
    num_trials = hier_cfg["test_num_obs_per_subject"]
    num_populations = hier_cfg["test_num_populations"]
    P = hier_cfg["num_params"]

    # Derived constants
    num_L_params = P + P * (P - 1) // 2
    num_pop_params = P + num_L_params

    # Prior hyperparameters from config
    prior_cfg = hier_cfg["prior"]
    lkj_concentration = float(prior_cfg["lkj_concentration"])
    halfnormal_scale = jnp.array(prior_cfg["halfnormal_scale"])
    mu_loc = jnp.array(prior_cfg["mu_loc"])
    mu_scale = jnp.array(prior_cfg["mu_scale"])

    # Build CholeskyLKJ-MVN prior distribution
    prior = create_hierarchical_rdm_prior_lkj_mvn(
        num_subjects,
        lkj_concentration=lkj_concentration,
        halfnormal_scale=halfnormal_scale,
        mu_loc=mu_loc,
        mu_scale=mu_scale,
    )

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

    # Create likelihood factories with correct num_pop_params
    likelihood_factory_approx = create_rdm_hierarchical_likelihood_factory_approx(
        conditioner, num_params=P, num_pop_params=num_pop_params,
    )

    mcmc_cfg = hier_cfg["mcmc"]

    def log_prior(x):
        """Log-prior on the flat MCMC vector using CholeskyLKJ-MVN prior.

        Uses HierarchicalRDMPriorLKJMVN.log_prob(L, mu, log_theta) which
        evaluates the joint distribution at (s, L, mu, log_theta) with
        s = sqrt(diag(L L^T)).  The ScaleMatvecDiag bijector inside the
        TransformedDistribution handles the L <-> rho_chol Jacobian.

        Only the flat -> L Jacobian (exp of log-diagonal) is added here.
        """
        sample = flat_to_prior_sample(x, num_subjects, P)
        mu = sample['mu']
        L = sample['L']
        log_theta = sample['log_theta']

        # Prior log-density in (L, mu, log_theta) space (includes L -> (rho_chol, s) Jacobian)
        lp = prior.log_prob(L, mu, log_theta).sum()

        # Jacobian: flat (log-diag) -> L (exp transform on diagonal)
        lp += jnp.sum(jnp.log(jnp.diag(L)))

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

        return trace.position

    # Generate and recover for each population
    test_key = jax.random.key(cfg["test_seed"])
    all_results = {}

    for pop_idx in range(num_populations):
        logger.info("Population %d / %d", pop_idx + 1, num_populations)

        test_key, data_key, sampling_key = jax.random.split(test_key, 3)

        # Simulate hierarchical data using LKJ-MVN prior
        data, context = sample_conditional_rdm_hierarchical_lkj_mvn(
            data_key, num_trials, num_subjects,
            lkj_concentration=lkj_concentration,
            halfnormal_scale=halfnormal_scale,
            mu_loc=mu_loc,
            mu_scale=mu_scale,
        )
        logger.info("Simulated data shape: %s", data.shape)

        # All subjects have the same trial count, mask is all True
        mask = jnp.ones((num_subjects, num_trials), dtype=bool)

        # Build initial position from true params
        init_position = context_to_flat(context)

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

        # Convert samples to interpretable form
        # samples_approx has shape (num_chains, num_samples, num_params)
        pop_results = {
            f"pop{pop_idx}_samples_approx": np.asarray(samples_approx),
            f"pop{pop_idx}_data": np.asarray(data),
            f"pop{pop_idx}_pop_mu_true": pop_params_true['mu'],
            f"pop{pop_idx}_pop_L_true": pop_params_true['L'],
            f"pop{pop_idx}_pop_Sigma_true": pop_params_true['Sigma'],
            f"pop{pop_idx}_pop_rho_true": pop_params_true['rho'],
            f"pop{pop_idx}_pop_s_true": pop_params_true['s'],
            f"pop{pop_idx}_subj_params_true": subj_params_true,
        }

        # Reference recovery (optional)
        if cfg["model"].get("run_reference_recovery", False):
            logger.info("Running reference recovery...")
            sampling_key, ref_key = jax.random.split(sampling_key)

            ref_likelihood = partial(
                create_rdm_hierarchical_likelihood,
                num_params=P, num_pop_params=num_pop_params,
            )
            samples_ref = recover_population(
                ref_key, data, mask, ref_likelihood, init_position,
            )
            samples_ref.block_until_ready()
            logger.info("Ref samples shape: %s", samples_ref.shape)
            pop_results[f"pop{pop_idx}_samples_ref"] = np.asarray(samples_ref)

        all_results.update(pop_results)

    logger.info("Saving results")
    save_hdf5("parameter_recovery_hierarchical.hdf5", all_results)


if __name__ == "__main__":
    main()
