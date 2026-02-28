
import logging
from pathlib import Path

import arviz.preview as az
import blackjax
import blackjax.smc.resampling as resampling
import hydra
from hydra.utils import instantiate
import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import numpy as np
import xarray as xr
from blackjax.smc import extend_params
from flax import nnx
from omegaconf import OmegaConf
from tensorflow_probability.substrates.jax import bijectors as tfb
from confrdm_jax.flows import load_conditioner
from confrdm_jax.flows import make_mlp_conditioner
from confrdm_jax.mcmc import warmup
from confrdm_jax.smc import smc_inference_loop

logger = logging.getLogger(__name__)

logging.getLogger("absl").setLevel(logging.ERROR)

jax.config.update('jax_enable_x64', True)

# Column names for observed data; RDM has 2 cols, CRDM has 3.
_DATA_COL_NAMES = ["rt", "choice", "condition"]


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


def _build_population_datatree(
    final_state,
    n_iter,
    data,
    context,
    log_theta_true,
    unravel_fn,
    bijector,
    num_params_ncp,
    num_subjects,
    param_names,
    pop_idx,
):
    """Reconstruct interpretable parameters and build an ArviZ-preview DataTree.

    Particles are stored in unconstrained flat space.  This function maps them
    back to constrained, exp-transformed parameters and assembles the groups
    required for ArviZ diagnostics:

    - ``posterior``    — population (mu, sigma) and subject-level parameters
    - ``sample_stats`` — SMC log-weights per particle per chain
    - ``observed_data``— RT, choice (and condition for CRDM) per subject/trial
    - ``constant_data``— true parameters for recovery assessment

    Args:
        final_state: SMC final state; ``.particles`` has shape
            ``(num_chains, num_particles, num_flat_params)`` and ``.weights``
            has shape ``(num_chains, num_particles)``.
        n_iter: Per-chain SMC iteration counts, shape ``(num_chains,)``.
        data: Observed data, shape ``(S, T, num_data_cols)``.
        context: Dict of true prior parameters from the sampler.
        log_theta_true: True subject log-parameters, shape ``(S, P)``.
        unravel_fn: Unravel function from ``jfu.ravel_pytree``.
        bijector: TFP JointMap bijector (constrained <-> unconstrained).
        num_params_ncp: Number of non-centered parameters.
        num_subjects: Number of subjects S.
        param_names: List of parameter names, length P.
        pop_idx: Population index, stored as a DataTree attribute.

    Returns:
        ``xr.DataTree`` with posterior, sample_stats, observed_data, and
        constant_data groups, ready to save as netCDF.
    """

    def reconstruct_single(flat_particle):
        """Map one flat unconstrained particle → (mu, s, log_theta)."""
        unconstrained = unravel_fn(flat_particle)
        params = bijector.forward(unconstrained)
        L = params['s'][:, None] * params['psi_raw']                    # (P, P)
        L_ncp = L[:num_params_ncp, :num_params_ncp]                     # (P_ncp, P_ncp)
        theta_ncp = params['mu'][:num_params_ncp] + jnp.einsum(
            'nj,ij->ni', params['z'], L_ncp,
        )                                                                # (S, P_ncp)
        log_theta = jnp.concatenate(
            [theta_ncp, params['theta_bt']], axis=-1,
        )                                                                # (S, P)
        return params['mu'], params['s'], log_theta

    # Vmap over particles (inner) then chains (outer):
    #   pop_mu:        (chains, draws, P)
    #   pop_s:         (chains, draws, P)
    #   pop_log_theta: (chains, draws, S, P)
    pop_mu, pop_s, pop_log_theta = jax.vmap(jax.vmap(reconstruct_single))(
        final_state.particles,
    )

    subjects = np.arange(num_subjects)

    posterior_ds = az.dict_to_dataset(
        {
            "mu":    np.exp(np.array(pop_mu)),   # population mean, original scale
            "sigma": np.array(pop_s),            # population SD, log-space
            # Subject-level parameters: (chains, draws, S)
            **{name: np.exp(np.array(pop_log_theta[..., i]))
               for i, name in enumerate(param_names)},
        },
        sample_dims=["chain", "draw"],
        coords={"subject": subjects, "param": param_names},
        dims={
            "mu":    ["param"],
            "sigma": ["param"],
            **{name: ["subject"] for name in param_names},
        },
    )

    sample_stats_ds = az.dict_to_dataset(
        {"log_weights": np.array(final_state.weights)},
        sample_dims=["chain", "draw"],
    )

    # Store all data columns (2 for RDM, 3 for CRDM)
    num_data_cols = data.shape[2]
    col_names = _DATA_COL_NAMES[:num_data_cols]
    observed_ds = xr.Dataset(
        {name: (["subject", "trial"], np.array(data[:, :, i]))
         for i, name in enumerate(col_names)},
        coords={"subject": subjects},
    )

    # True parameters for recovery assessment (not random, not observed → constant_data)
    constant_ds = xr.Dataset(
        {
            "log_theta": (["subject", "param"], np.array(log_theta_true)),
            "theta":     (["subject", "param"], np.array(jnp.exp(log_theta_true))),
            "mu":        (["param"], np.array(context["mu"])),
            "sigma":     (["param"], np.array(context["s"])),
        },
        coords={"subject": subjects, "param": param_names},
    )

    dt = xr.DataTree.from_dict({
        "posterior":    posterior_ds,
        "sample_stats": sample_stats_ds,
        "observed_data": observed_ds,
        "constant_data": constant_ds,
    })

    # Store scalar/per-chain metadata as attributes
    dt.attrs["smc_n_iter"]    = np.array(n_iter).tolist()
    dt.attrs["pop_idx"]       = pop_idx
    dt.attrs["num_subjects"]  = num_subjects
    dt.attrs["param_names"]   = param_names

    return dt


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
    num_params_ncp = num_params - 2  # Non-centered for all except b and t0
    num_L_params = num_params + num_params * (num_params - 1) // 2

    param_names = model_hier_cfg["param_names"]

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
    conditioner_key, _ = jax.random.split(train_key)

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
        "s": tfb.Exp(),                       # InverseGamma -> Positive
        "mu": tfb.Identity(),                 # Normal -> Real
        "psi_raw": tfb.CorrelationCholesky(), # CholeskyLKJ -> unconstrained vector
        "z": tfb.Identity(),                  # N(0,I) -> Real (already unconstrained)
        "theta_bt": tfb.Identity(),           # log(b), log(t0) -> Real
    })

    def recover_population(sampling_key, data, mask, create_likelihood_fun, init_position, unravel_fn):
        """Run tempered SMC to recover hierarchical parameters for one population."""

        # log_prior_fn defined here so it can close over the per-population unravel_fn.
        # Jacobian components are computed separately to avoid a broadcasting bug in
        # JointMap.forward_log_det_jacobian (CC Jacobian gets broadcast across the
        # Exp Jacobian's shape-(P,) output, overcounting by P).
        def log_prior_fn(flat_params):
            unconstrained_params_dict = unravel_fn(flat_params)
            params = bijector.forward(unconstrained_params_dict)
            prior_lp = prior.log_prob(params)
            ljc_exp = jnp.sum(unconstrained_params_dict['s'])  # Exp bijector: sum(log(s))
            ljc_cc = tfb.CorrelationCholesky().forward_log_det_jacobian(
                unconstrained_params_dict['psi_raw'],
                event_ndims=1,  # input is a 1-D unconstrained vector
            )
            return prior_lp + ljc_exp + ljc_cc

        likelihood_fun = create_likelihood_fun(data, mask)

        # Reconstruct theta from NCP (z) and centered (theta_bt)
        def log_likelihood_fn_wrapped(flat_params):
            unconstrained_params_dict = unravel_fn(flat_params)
            params = bijector.forward(unconstrained_params_dict)
            L = params['s'][:, None] * params['psi_raw']          # (P, P)
            L_ncp = L[:num_params_ncp, :num_params_ncp]           # (P_ncp, P_ncp)
            theta_ncp = params['mu'][:num_params_ncp] + jnp.einsum('nj,ij->ni', params['z'], L_ncp)  # (S, P_ncp)
            theta = jnp.concatenate([theta_ncp, params['theta_bt']], axis=-1)  # (S, P)
            return likelihood_fun(theta)

        def logdensity_fn(params):
            return log_prior_fn(params) + log_likelihood_fn_wrapped(params)

        # Run window adaptation to find good HMC parameters.
        # target_acceptance_rate=0.9 recommended for hierarchical models with
        # non-trivial geometry near the Cholesky constraint boundary.
        sampling_key, warmup_key = jax.random.split(sampling_key)
        _, _, adapted_params = warmup(
            blackjax.nuts,
            logdensity_fn,  # Requires logdensity = logprior + loglikelihood
            init_position,
            smc_cfg["num_warmup"],
            warmup_key,
            target_acceptance_rate=0.9,
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

        # Reconstruct interpretable subject-level parameters from prior samples
        L_true = context['s'][:, None] * context['psi_raw']                  # (P, P)
        L_ncp_true = L_true[:num_params_ncp, :num_params_ncp]                # (P_ncp, P_ncp)
        theta_ncp_true = context['mu'][:num_params_ncp] + jnp.einsum(
            'nj,ij->ni', context['z'], L_ncp_true,
        )                                                                      # (S, P_ncp)
        log_theta_true = jnp.concatenate(
            [theta_ncp_true, context['theta_bt']], axis=-1,
        )                                                                      # (S, P)

        # Approximate recovery
        logger.info("Running approximate recovery...")
        sampling_key, approx_key = jax.random.split(sampling_key)

        final_state, n_iter = recover_population(
            approx_key, data, mask, likelihood_factory_approx, init_position, unravel_fn,
        )
        logger.info("SMC converged in %s iterations", n_iter)

        dt = _build_population_datatree(
            final_state=final_state,
            n_iter=n_iter,
            data=data,
            context=context,
            log_theta_true=log_theta_true,
            unravel_fn=unravel_fn,
            bijector=bijector,
            num_params_ncp=num_params_ncp,
            num_subjects=num_subjects,
            param_names=param_names,
            pop_idx=pop_idx,
        )
        approx_path = f"hierarchical_recovery_pop{pop_idx}_approx.nc"
        logger.info("Saving approx results to %s", approx_path)
        dt.to_netcdf(approx_path)

        # Reference recovery (optional)
        if cfg["model"].get("run_reference_recovery", False):
            logger.info("Running reference recovery...")
            sampling_key, ref_key = jax.random.split(sampling_key)

            ref_likelihood = instantiate(model_hier_cfg["likelihood_factory_ref"])
            final_state_ref, n_iter_ref = recover_population(
                ref_key, data, mask, ref_likelihood, init_position, unravel_fn,
            )
            logger.info("Ref SMC converged in %s iterations", n_iter_ref)

            dt_ref = _build_population_datatree(
                final_state=final_state_ref,
                n_iter=n_iter_ref,
                data=data,
                context=context,
                log_theta_true=log_theta_true,
                unravel_fn=unravel_fn,
                bijector=bijector,
                num_params_ncp=num_params_ncp,
                num_subjects=num_subjects,
                param_names=param_names,
                pop_idx=pop_idx,
            )
            ref_path = f"hierarchical_recovery_pop{pop_idx}_ref.nc"
            logger.info("Saving ref results to %s", ref_path)
            dt_ref.to_netcdf(ref_path)


if __name__ == "__main__":
    main()
