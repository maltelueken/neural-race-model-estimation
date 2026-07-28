"""Hierarchical parameter recovery for a trained conditioner.

Simulates whole *populations* from the hierarchical LKJ-MVN prior and fits each
one as a single joint model over population-level and subject-level parameters
at once.  Unlike ``parameter_recovery.py``, where each data set is an
independent fit, here the subjects are tied together and shrinkage is part of
what is being recovered.

Inference is tempered SMC rather than NUTS.  The semi-centered hierarchical
posterior has strong funnel geometry, and annealing a particle cloud from the
prior copes with that better than a single chain; the cost is that the stored
draws are particles, so what is called a "chain" here is an independent SMC
run rather than a Markov chain.

Sampling happens in a flat *unconstrained* space: prior draws are pushed
through ``bijector.inverse`` and ravelled to a vector, and mapped back for
evaluation.  Two places therefore have to agree with the prior class about the
semi-centered reconstruction — ``log_likelihood_fn_wrapped`` during sampling
and ``_reconstruct_particle`` afterwards.

Must be launched with the same overrides that produced the checkpoint::

    python scripts/parameter_recovery_hierarchical.py model=rdm

Writes ``hierarchical_recovery_pop{N}_approx.nc`` per population.
"""

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
from confrdm_jax.mcmc import warmup_multiple_chains
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


def _reconstruct_particle(flat_particle, unravel_fn, bijector, num_params_ncp):
    """Map one flat unconstrained particle → (mu, s, log_theta).

    Undoes the flatten-and-unconstrain that ``sample_prior_particles`` applies,
    then rebuilds subject-level log-parameters from the semi-centered
    parameterisation: the leading `num_params_ncp` parameters come from the
    standard-normal offsets ``z`` scaled by the covariance Cholesky, the
    trailing ones are already on the parameter scale in ``theta_bt``.

    This is the same reconstruction as ``log_likelihood_fn_wrapped`` and as the
    hierarchical simulators; all of them must agree or the recovered parameters
    will not be the ones the data were generated from.

    Returns:
        ``(mu, s, log_theta)`` with shapes ``(P,)``, ``(P,)`` and ``(S, P)``,
        all still in **log** space for the parameters themselves.
    """
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


def _build_population_datatree(
    particles,
    weights,
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
    - ``sample_stats`` — final-increment SMC weights per particle per chain
    - ``observed_data``— RT, choice (and condition for CRDM) per subject/trial
    - ``constant_data``— true parameters for recovery assessment

    Warning:
        ``mu`` is **not on the same scale in both groups**. The posterior
        stores ``exp(mu)`` (natural scale, matching the subject-level
        variables) while ``constant_data`` stores raw log-space ``mu``, so
        anything comparing the two must exponentiate the truth first —
        ``notebooks/create_figures_parameter_recovery_hierarchical.ipynb``
        does exactly that. ``sigma`` is log-space in both and needs no such
        correction, since it is a standard deviation *of* log-parameters.
        Subject-level parameters are the well-behaved case: ``constant_data``
        carries both ``log_theta`` and ``theta``, named for their scales.
        Changing this is a file-format break — the notebook's compensating
        ``exp`` would then double-apply — so it is documented rather than
        fixed.

    Args:
        particles: Resampled SMC particles, equally weighted, shape
            ``(num_chains, num_particles, num_flat_params)``.
        weights: Final-increment weights, shape ``(num_chains, num_particles)``.
            **Diagnostic only** — they do not index `particles`, which have
            already been resampled against them. These are normalised *linear*
            weights, not logs. Their ESS says how far from uniform the last
            tempering step left the cloud, and correlates at −0.96 with how far
            ignoring the weights would have moved the posterior — so it is the
            per-run indicator of whether the final resample mattered.
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

    reconstruct = lambda p: _reconstruct_particle(p, unravel_fn, bijector, num_params_ncp)

    # Vmap over particles (inner) then chains (outer):
    #   pop_mu:        (chains, draws, P)
    #   pop_s:         (chains, draws, P)
    #   pop_log_theta: (chains, draws, S, P)
    pop_mu, pop_s, pop_log_theta = jax.vmap(jax.vmap(reconstruct))(particles)

    subjects = np.arange(num_subjects)

    posterior_ds = az.dict_to_dataset(
        {
            # exp(mu): the population location on the natural scale (the
            # median of the lognormal, not its mean).  NB constant_data["mu"]
            # is *not* exponentiated — see the warning above.
            "mu":    np.exp(np.array(pop_mu)),
            # Between-subject SD of the log-parameters; stays in log space.
            "sigma": np.array(pop_s),
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

    # Named `weights`, not `log_weights`: they are normalised linear weights.
    sample_stats_ds = az.dict_to_dataset(
        {"weights": np.array(weights)},
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
            # Log space, unlike posterior["mu"] — see the warning above.
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

    def recover_population(sampling_key, data, mask, create_likelihood_fun, unravel_fn):
        """Run tempered SMC to recover hierarchical parameters for one population.

        Returns:
            ``(particles, weights, n_iter, initial_particles)``. `particles` are
            resampled against the final weights and so are equally weighted;
            `weights` are retained as a diagnostic only. All four carry a
            leading `num_chains` axis.

        Chains run through ``jax.lax.map``, not ``vmap``, so they are
        sequential and `num_chains` multiplies wall-clock time linearly. That
        is also why giving each chain its own particle cloud costs nothing.
        """

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

        num_chains = smc_cfg["num_chains"]

        sampling_key, init_key, warmup_key, cloud_key, chain_key = jax.random.split(
            sampling_key, 5,
        )

        # Overdispersed warm-up starts: one prior draw per chain, already in the
        # unconstrained flat space the sampler works in.  Starting every chain
        # from `prior.mode()` made the adaptation — and therefore the mutation
        # kernel every chain shares — a single draw with no variability at all.
        init_positions = sample_prior_particles(
            prior, bijector, num_chains, init_key,
        )

        # One window adaptation per chain.  target_acceptance_rate=0.8 avoids the
        # degenerate near-zero step sizes that 0.9 produces when the NLE
        # posterior has high curvature.  Note the adaptation targets the
        # lambda = 1 posterior but the step size is used from lambda = 0
        # onward, where the target is the much broader prior — the degenerate
        # step-size guard below is a symptom of that mismatch.
        _, adapted_params = warmup_multiple_chains(
            blackjax.nuts,
            logdensity_fn,  # Requires logdensity = logprior + loglikelihood
            init_positions,
            smc_cfg["num_warmup"],
            warmup_key,
            target_acceptance_rate=0.8,
        )

        # Chains adapt independently, so the degenerate-step-size repair has to
        # be a per-chain select rather than the scalar Python branch it was.
        step_sizes = adapted_params["step_size"]
        degenerate = step_sizes < 1e-4
        if bool(jnp.any(degenerate)):
            logger.warning(
                "Degenerate step size in %d/%d chains after warmup — overriding to 1e-3",
                int(jnp.sum(degenerate)), num_chains,
            )
        step_sizes = jnp.where(degenerate, 1e-3, step_sizes)
        logger.info("Adapted step sizes per chain: %s", np.asarray(step_sizes))

        # One prior cloud per chain.  Sharing a single cloud left any gap in that
        # one draw invisible to every between-chain comparison.
        initial_particles = jax.vmap(
            lambda key: sample_prior_particles(
                prior, bijector, smc_cfg["num_particles"], key,
            ),
        )(jax.random.split(cloud_key, num_chains))

        def run_chain(key, step_size, inverse_mass_matrix, particles):
            """One SMC run with this chain's own tuning and its own particles."""
            run_key, resample_key = jax.random.split(key)
            tempered = blackjax.adaptive_tempered_smc(
                log_prior_fn,
                log_likelihood_fn_wrapped, # Requires only loglikelihood
                blackjax.hmc.build_kernel(),
                blackjax.hmc.init,
                extend_params(dict(
                    step_size=step_size,
                    inverse_mass_matrix=inverse_mass_matrix,
                    num_integration_steps=smc_cfg["num_integration_steps"],
                )),
                resampling.systematic,
                smc_cfg["target_ess"],
                num_mcmc_steps=smc_cfg["num_mcmc_steps"],
            )
            n_iter, state = smc_inference_loop(run_key, tempered.step, tempered.init(particles))

            # Each SMC step is resample -> mutate -> reweight, so the weights on
            # the returned state belong to the final temperature increment and
            # were never resampled away.  Storing the particles as if they were
            # equally weighted therefore reports the *penultimate* tempered
            # target, which is flatter than the posterior — an over-dispersion
            # bias, not just noise (measured over 7480 scalar quantities from
            # 15 stored runs: weighted SD below raw SD in 80.2% of them,
            # per-chain means off by up to 0.23 posterior SD).  One systematic
            # resample makes the stored draws genuinely uniform.
            #
            # Resampling beats storing the weights and applying them
            # downstream: ArviZ has no weighted-posterior support, so every
            # notebook, R-hat and ESS computation would otherwise have to
            # handle weights itself.  The cost is ordinary resampling noise,
            # far smaller than the bias removed at the observed weight ESS
            # (min 61%, median 91% of num_particles).
            idx = resampling.systematic(
                resample_key, state.weights, smc_cfg["num_particles"],
            )
            return n_iter, state.particles[idx], state.weights

        # `weights` are kept only as a diagnostic: their ESS says how much the
        # resample above actually did.  They no longer index `particles`.
        n_iter, particles, weights = jax.lax.map(
            lambda xs: run_chain(*xs),
            (
                jax.random.split(chain_key, num_chains),
                step_sizes,
                adapted_params["inverse_mass_matrix"],
                initial_particles,
            ),
        )

        weight_ess = 1.0 / jnp.sum(weights ** 2, axis=-1) / smc_cfg["num_particles"]
        logger.info("Final-increment weight ESS per chain: %s",
                    np.round(np.asarray(weight_ess), 3))

        return particles, weights, n_iter, initial_particles

    # Generate and recover for each population
    test_key = jax.random.key(cfg["test_seed"])

    # Compute unravel_fn once (shape is the same for all populations).  The
    # flattened mode itself is unused — warm-up starts from per-chain prior
    # draws, not from a single point.
    _, unravel_fn = jfu.ravel_pytree(bijector.inverse(prior.mode()))

    reconstruct = lambda p: _reconstruct_particle(p, unravel_fn, bijector, num_params_ncp)

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

        particles, weights, n_iter, initial_particles = recover_population(
            approx_key, data, mask, likelihood_factory_approx, unravel_fn,
        )
        logger.info("SMC converged in %s iterations", n_iter)

        # Each chain now has its own cloud, shape (chains, particles, params).
        # The prior group pools them: they are all draws from the same prior.
        pooled_particles = initial_particles.reshape(-1, initial_particles.shape[-1])
        prior_mu, prior_s, prior_log_theta = jax.vmap(reconstruct)(pooled_particles)
        subjects = np.arange(num_subjects)
        prior_ds = xr.Dataset(
            {
                "mu":    (["draw", "param"], np.exp(np.array(prior_mu))),
                "sigma": (["draw", "param"], np.array(prior_s)),
                **{name: (["draw", "subject"], np.exp(np.array(prior_log_theta[..., i])))
                   for i, name in enumerate(param_names)},
            },
            coords={
                "draw": np.arange(pooled_particles.shape[0]),
                "subject": subjects,
                "param": param_names,
            },
        )

        dt = _build_population_datatree(
            particles=particles,
            weights=weights,
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
        dt["prior"] = prior_ds
        approx_path = f"hierarchical_recovery_pop{pop_idx}_approx.nc"
        logger.info("Saving approx results to %s", approx_path)
        dt.to_netcdf(approx_path)

        # Reference recovery (optional)
        if cfg["model"].get("run_reference_recovery", False):
            logger.info("Running reference recovery...")
            sampling_key, ref_key = jax.random.split(sampling_key)

            ref_likelihood = instantiate(model_hier_cfg["likelihood_factory_ref"])
            particles_ref, weights_ref, n_iter_ref, _ = recover_population(
                ref_key, data, mask, ref_likelihood, unravel_fn,
            )
            logger.info("Ref SMC converged in %s iterations", n_iter_ref)

            dt_ref = _build_population_datatree(
                particles=particles_ref,
                weights=weights_ref,
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
            dt_ref["prior"] = xr.DataTree(dataset=prior_ds)
            ref_path = f"hierarchical_recovery_pop{pop_idx}_ref.nc"
            logger.info("Saving ref results to %s", ref_path)
            dt_ref.to_netcdf(ref_path)


if __name__ == "__main__":
    main()
