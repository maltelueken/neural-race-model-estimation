"""Hierarchical parameter recovery for a trained conditioner.

Simulates whole *populations* from the hierarchical LKJ-MVN prior and fits each one as a
single joint model over population-level and subject-level parameters at once. Unlike
``parameter_recovery.py``, where each data set is an independent fit, here the subjects are
tied together and shrinkage is part of what is being recovered.

Inference is tempered SMC rather than NUTS. The semi-centered hierarchical posterior has
strong funnel geometry, and annealing a particle cloud from the prior copes with that better
than a single chain; the cost is that the stored draws are particles, so what is called a
"chain" here is an independent SMC run rather than a Markov chain.

Sampling happens in a flat *unconstrained* space, owned by
:class:`eamax.hierarchical.HierarchicalFlatSpace`: it maps between the sampler's vector and
the prior's five named components, applies the change of variables, and runs the
semi-centered reconstruction. That last one used to be written out in four places — the
simulators, the likelihood wrapper, the particle post-processing and the truth
reconstruction — all of which had to agree or the recovered parameters would not be the ones
the data were generated from. Now there is one definition.

**What each fit does, in order.** Draw one independent particle cloud per chain from the
prior conditioned on ``t0 < 0.97 * min(rt)`` per subject; adapt each chain's mutation kernel
on its own with window adaptation; drop any chain whose adaptation collapsed; temper the
clouds to the posterior. No tuning is substituted or repaired at any point — see
:data:`DEGENERATE_STEP_SIZE`.

Must be launched with the same overrides that produced the checkpoint::

    python scripts/parameter_recovery_hierarchical.py model=rdm

Writes ``hierarchical_recovery_pop{N}_approx.nc`` per population.
"""

import logging
from pathlib import Path

import arviz as az
import blackjax
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from eamax.hierarchical import reconstruct_from_dict
from eamax.inference import (
    T0Support,
    init_particles_from_prior,
    min_valid_rt,
    tempered_smc,
    window_adaptation,
)
from flax import nnx
from hydra.utils import instantiate
from omegaconf import OmegaConf

from confrdm_jax import configure_jax
from confrdm_jax.flows_affine import load_conditioner, make_mlp_conditioner
from confrdm_jax.specs import DATA_COL_NAMES, spec_for

logger = logging.getLogger(__name__)

logging.getLogger("absl").setLevel(logging.ERROR)

#: Below this, window adaptation has collapsed rather than converged: healthy chains on these
#: posteriors land at 0.06-0.16, so anything here is two to three orders of magnitude out and
#: the chain would barely move during mutation.
#:
#: Such a chain is **dropped and reported, never repaired**. Substituting the healthy chains'
#: median step size and mass matrix was tried and measured: the repaired chains came out
#: under-dispersed at 0.16-0.83x their siblings' spread and still broke R-hat
#: (1.94 / 2.60 / 2.76), falling to ~1.01 only once they were dropped — the step-size column
#: looked fixed, the fit was not. The cause is a start outside the ``t0`` support, which
#: :class:`eamax.inference.init.T0Support` now removes at the source; a collapse that
#: survives that is a finding about the posterior and belongs in the run's output.
DEGENERATE_STEP_SIZE = 1e-4


def healthy_chains(step_sizes):
    """Which chains adapted, and how many did not.

    Split out from the fit so the drop rule — the one piece of the migration most worth
    getting right — is testable without running SMC.

    Parameters
    ----------
    step_sizes : array
        Adapted step size per chain, shape ``(num_chains,)``.

    Returns
    -------
    kept : numpy.ndarray
        Indices of the chains to sample with, in order.
    num_dropped : int
        How many were discarded. **Zero when every chain collapsed** — there is then nothing
        to compare against and nothing to keep, so all are kept and the caller reports the
        run as unusable rather than returning an empty result.
    """
    step_sizes = np.asarray(step_sizes)
    kept = np.flatnonzero(step_sizes >= DEGENERATE_STEP_SIZE)
    if kept.size == 0:
        return np.arange(step_sizes.size), 0
    return kept, int(step_sizes.size - kept.size)


def _particle_dataset(flat_particles, flat_space, param_names):
    """Reconstruct a flat particle cloud into a ``(draw, ...)`` Dataset.

    Shared by the ``prior`` and ``smc_initial_particles`` groups, which differ only in which
    cloud they are given. Scales match the ``posterior`` group's convention: ``mu`` and the
    subject-level parameters are exponentiated to the natural scale, ``sigma`` stays in log
    space because it is a standard deviation *of* log-parameters.

    Args:
        flat_particles: ``(N, D)`` unconstrained flat particles.
        flat_space: The coordinate system they live in.
        param_names: Parameter names, length P.

    Returns:
        ``xr.Dataset`` with dims ``draw``, ``param`` and ``subject``.
    """
    def one(flat):
        mu, s = flat_space.population(flat)
        return mu, s, flat_space.subject_params(flat)

    mu, s, log_theta = jax.vmap(one)(flat_particles)
    num_subjects = log_theta.shape[1]

    return xr.Dataset(
        {
            "mu":    (["draw", "param"], np.exp(np.array(mu))),
            "sigma": (["draw", "param"], np.array(s)),
            **{name: (["draw", "subject"], np.exp(np.array(log_theta[..., i])))
               for i, name in enumerate(param_names)},
        },
        coords={
            "draw": np.arange(flat_particles.shape[0]),
            "subject": np.arange(num_subjects),
            "param": list(param_names),
        },
    )


def _flatten_diagnostic(ds):
    """Flatten a per-variable diagnostic Dataset into ``(value, label)`` pairs.

    ``az.rhat`` and ``az.ess`` return one value per variable *per coordinate* — `mu` and
    `sigma` carry a ``param`` axis, the subject-level parameters carry a ``subject`` axis —
    so the worst entry has to be found across a ragged collection rather than a single array.
    Labels name where the value came from, which is the part worth logging: "3.74 at t0[17]"
    localises the failure, "3.74" does not.

    Non-finite entries are dropped. R-hat is NaN for a single chain, so a one-chain run
    yields an empty list rather than a spurious extreme.
    """
    pairs = []
    for name, da in ds.items():
        if not da.dims:
            pairs.append((float(da.values), name))
            continue
        stacked = da.stack(entry=da.dims)
        for coord, value in zip(
            stacked["entry"].values, np.asarray(stacked.values), strict=True,
        ):
            coords = coord if isinstance(coord, tuple) else (coord,)
            pairs.append((float(value), f"{name}[{','.join(str(c) for c in coords)}]"))
    return [(value, label) for value, label in pairs if np.isfinite(value)]


def _log_convergence(dt, label, max_rhat, min_ess):
    """Check split-R-hat and ESS on the posterior, log them, and record them.

    Runs before the ``.nc`` is written so a bad population is visible during the run rather
    than in the notebook days later — past runs recorded R-hat up to 2.76 and 9.42, and
    nothing in the script noticed at the time.

    R-hat is the diagnostic that carries weight here. Each "chain" is an independent SMC run
    with its own starting cloud and its own adapted mutation kernel, so between-chain
    disagreement means what Gelman-Rubin assumes it means.

    **ESS is weaker than it looks for SMC** and is logged for continuity with the MCMC path
    rather than as a guarantee. It is an autocorrelation estimate, and particles within a
    chain are exchangeable — their storage order is arbitrary — so a cloud that resampling
    has collapsed onto a few distinct values still reports an ESS near the particle count.
    Duplicated particles are what ``smc_num_unique`` measures; read the two together.
    """
    posterior = dt["posterior"].to_dataset()

    rhat_pairs = _flatten_diagnostic(az.rhat(posterior))
    ess_pairs = _flatten_diagnostic(az.ess(posterior))

    if rhat_pairs:
        worst_rhat, rhat_at = max(rhat_pairs)
        dt.attrs["max_rhat"] = worst_rhat
        dt.attrs["max_rhat_at"] = rhat_at
    else:
        worst_rhat, rhat_at = float("nan"), "n/a (needs >= 2 chains)"

    worst_ess, ess_at = min(ess_pairs) if ess_pairs else (float("nan"), "n/a")
    if ess_pairs:
        dt.attrs["min_ess"] = worst_ess
        dt.attrs["min_ess_at"] = ess_at

    logger.info(
        "%s convergence: max R-hat %.3f at %s, min ESS %.0f at %s",
        label, worst_rhat, rhat_at, worst_ess, ess_at,
    )

    if rhat_pairs and worst_rhat > max_rhat:
        logger.warning(
            "%s: max R-hat %.3f at %s exceeds %.3f — the chains disagree, so the pooled "
            "posterior in this file is not a single distribution. Inspect per-chain draws "
            "before using it.",
            label, worst_rhat, rhat_at, max_rhat,
        )
    if ess_pairs and worst_ess < min_ess:
        logger.warning(
            "%s: min ESS %.0f at %s is below %.0f — quantiles and R-hat on that quantity are "
            "themselves too noisy to trust.",
            label, worst_ess, ess_at, min_ess,
        )


def _build_population_datatree(
    result, flat_space, data, context, log_theta_true, param_names, pop_idx, num_dropped,
):
    """Reconstruct interpretable parameters and build an ArviZ-preview DataTree.

    Particles are stored in unconstrained flat space. This maps them back to constrained,
    exp-transformed parameters through `flat_space` — the same semi-centered reconstruction
    the sampler used — and assembles the groups ArviZ diagnostics need:

    - ``posterior``     — population (mu, sigma) and subject-level parameters
    - ``sample_stats``  — per-particle SMC weights
    - ``observed_data`` — RT, choice (and condition for CRDM) per subject/trial
    - ``constant_data`` — true parameters for recovery assessment

    Warning:
        ``mu`` is **not on the same scale in both groups**. The posterior stores ``exp(mu)``
        (natural scale, matching the subject-level variables) while ``constant_data`` stores
        raw log-space ``mu``, so anything comparing the two must exponentiate the truth first
        — ``notebooks/create_figures_parameter_recovery_hierarchical.ipynb`` does exactly
        that. ``sigma`` is log-space in both and needs no such correction, since it is a
        standard deviation *of* log-parameters. Subject-level parameters are the well-behaved
        case: ``constant_data`` carries both ``log_theta`` and ``theta``, named for their
        scales. Changing this is a file-format break — the notebook's compensating ``exp``
        would then double-apply — so it is documented rather than fixed.

    Args:
        result: :class:`eamax.inference.smc.SMCResult`; every field carries a leading chain
            axis. Its ``particles`` have been resampled against the final weights, so
            ``weights`` come back uniform and are stored for shape compatibility only; the
            informative pre-resample figure is ``weight_ess``.
        flat_space: The coordinate system the particles live in.
        data: Observed data, shape ``(S, T, num_data_cols)``.
        context: The prior draw the data were generated from.
        log_theta_true: True subject log-parameters, shape ``(S, P)``.
        param_names: Parameter names, length P.
        pop_idx: Population index, stored as an attribute.
        num_dropped: Chains dropped for a collapsed step size before sampling.

    Returns:
        ``xr.DataTree`` ready to save as netCDF.
    """
    def one(flat):
        mu, s = flat_space.population(flat)
        return mu, s, flat_space.subject_params(flat)

    # Vmap over particles (inner) then chains (outer):
    #   pop_mu / pop_s: (chains, draws, P);  pop_log_theta: (chains, draws, S, P)
    pop_mu, pop_s, pop_log_theta = jax.vmap(jax.vmap(one))(result.particles)

    num_subjects = data.shape[0]
    subjects = np.arange(num_subjects)

    posterior_ds = az.dict_to_dataset(
        {
            # exp(mu): the population location on the natural scale (the median of the
            # lognormal, not its mean). NB constant_data["mu"] is *not* exponentiated.
            "mu":    np.exp(np.array(pop_mu)),
            # Between-subject SD of the log-parameters; stays in log space.
            "sigma": np.array(pop_s),
            **{name: np.exp(np.array(pop_log_theta[..., i]))
               for i, name in enumerate(param_names)},
        },
        sample_dims=["chain", "draw"],
        coords={"subject": subjects, "param": list(param_names)},
        dims={
            "mu":    ["param"],
            "sigma": ["param"],
            **{name: ["subject"] for name in param_names},
        },
    )

    # Named `weights`, not `log_weights`: they are normalised linear weights, and uniform
    # because the final resample has already spent them.
    sample_stats_ds = az.dict_to_dataset(
        {"weights": np.array(result.weights)}, sample_dims=["chain", "draw"],
    )

    col_names = list(DATA_COL_NAMES[:data.shape[2]])
    observed_ds = xr.Dataset(
        {name: (["subject", "trial"], np.array(data[:, :, i]))
         for i, name in enumerate(col_names)},
        coords={"subject": subjects},
    )

    # True parameters for recovery assessment (not random, not observed -> constant_data)
    constant_ds = xr.Dataset(
        {
            "log_theta": (["subject", "param"], np.array(log_theta_true)),
            "theta":     (["subject", "param"], np.array(jnp.exp(log_theta_true))),
            # Log space, unlike posterior["mu"] — see the warning above.
            "mu":        (["param"], np.array(context["mu"])),
            "sigma":     (["param"], np.array(context["s"])),
        },
        coords={"subject": subjects, "param": list(param_names)},
    )

    dt = xr.DataTree.from_dict({
        "posterior":     posterior_ds,
        "sample_stats":  sample_stats_ds,
        "observed_data": observed_ds,
        "constant_data": constant_ds,
    })

    dt.attrs["smc_n_iter"] = np.array(result.num_iterations).tolist()
    # Read these next to `posterior`'s draw count: any shortfall is the factor by which the
    # stored ESS is optimistic.
    dt.attrs["smc_num_unique"] = np.array(result.num_unique).tolist()
    dt.attrs["smc_num_unique_pre_resample"] = np.array(result.num_unique_smc).tolist()
    # Weight ESS as a fraction, of the cloud *before* the final resample — it says how much
    # that resample actually did. The stored weights are uniform, so their own ESS is 1 by
    # construction and measures nothing.
    dt.attrs["smc_weight_ess"] = np.array(result.weight_ess).tolist()
    # The SMC estimate of log p(data | model), accumulated for free by the tempering loop.
    # The spread across chains is the honest standard error, and is only meaningful because
    # each chain got an independent cloud and its own adapted kernel.
    dt.attrs["log_marginal_likelihood"] = np.array(result.log_marginal_likelihood).tolist()
    dt.attrs["log_marginal_likelihood_mean"] = float(np.mean(result.log_marginal_likelihood))
    dt.attrs["num_dropped_chains"] = int(num_dropped)
    dt.attrs["pop_idx"] = pop_idx
    dt.attrs["num_subjects"] = num_subjects
    dt.attrs["param_names"] = list(param_names)

    return dt


@hydra.main(version_base=None, config_path="../conf_jax", config_name="config")
def main(cfg):
    configure_jax(cfg["device"], require_device=cfg["require_device"])

    hier_cfg = cfg["hierarchical_recovery"]
    model_cfg = cfg["model"]
    model_hier_cfg = model_cfg["hierarchical"]
    smc_cfg = hier_cfg["smc"]
    init_cfg = cfg["init"]

    num_subjects = hier_cfg["num_subjects"]
    num_trials = hier_cfg["test_num_obs_per_subject"]
    num_populations = hier_cfg["test_num_populations"]
    num_chains = smc_cfg["num_chains"]
    num_particles = smc_cfg["num_particles"]

    spec = spec_for(model_cfg["spec"])
    param_names = list(spec.names)
    context_names = list(model_cfg["context_names"])

    # Prior hyperparameters from model-specific config
    prior_cfg = OmegaConf.to_container(model_hier_cfg["prior"], resolve=True)

    prior_factory = instantiate(model_hier_cfg["prior_factory"])
    sampler = instantiate(model_hier_cfg["test_sampler"])
    likelihood_factory_fn = instantiate(model_hier_cfg["likelihood_factory_approx"])

    prior = prior_factory(num_subjects, **prior_cfg)
    # One flat unconstrained coordinate system, shared by the sampler, the log-prior (which
    # carries the change of variables) and the post-processing.
    flat_space = prior.flat_space()
    logger.info(
        "Hierarchical prior: %s, %d flat parameters", prior, flat_space.num_flat_params,
    )

    train_key = jax.random.key(cfg["train_seed"])
    conditioner_key, _ = jax.random.split(train_key)

    conditioner = make_mlp_conditioner(
        num_in=len(context_names),
        num_bins=model_cfg["num_bins"],
        num_mid=model_cfg["num_mid"],
        affine=model_cfg["flow_affine"],
        rngs=nnx.Rngs(default=conditioner_key),
    )

    conditioner_path = Path(cfg["conditioner_dir"]).absolute() / "conditioner"
    logger.info("Loading conditioner from: %s", conditioner_path)
    conditioner = load_conditioner(conditioner, conditioner_path, context_names=context_names)
    conditioner.eval()

    likelihood_factory_approx = likelihood_factory_fn(conditioner)

    def recover_population(sampling_key, data, mask, create_likelihood_fun, min_rt):
        """Run tempered SMC to recover hierarchical parameters for one population.

        Returns:
            ``(result, initial_particles, num_dropped)``. `result` is an
            :class:`eamax.inference.smc.SMCResult` over the chains that survived the
            step-size check; `initial_particles` is their starting cloud; `num_dropped` is
            how many chains were discarded for a collapsed adaptation.
        """
        likelihood_fun = create_likelihood_fun(data, mask)

        def log_likelihood_fn(flat):
            return likelihood_fun(flat_space.subject_params(flat))

        # `flat_space.log_prob` scores the prior in flat unconstrained coordinates, change of
        # variables included. The Jacobian is reduced component by component with each
        # component's own event rank; calling `JointMap.forward_log_det_jacobian` without
        # `event_ndims` overcounts the CorrelationCholesky term by (P-1) times a
        # state-dependent quantity, which tilts the posterior over correlations rather than
        # cancelling.
        def logdensity_fn(flat):
            return flat_space.log_prob(flat) + log_likelihood_fn(flat)

        init_key, warmup_key, cloud_key, chain_key = jax.random.split(sampling_key, 4)

        # Per-subject support bound for `t0`, with the censoring sentinel excluded. Per
        # subject, not pooled: `t0` is a subject-level parameter, so pooling would let a fast
        # subject's floor license a start above a slow subject's fastest trial.
        support = T0Support.from_spec(
            spec, min_rt, max_fraction=init_cfg["t0_max_fraction"],
        )
        max_attempts = init_cfg["t0_rejection_max_attempts"]

        # Overdispersed, support-aware warm-up starts: one prior draw per chain, already in
        # the flat space the sampler works in. A raw prior draw is what used to put a chain on
        # the `t0 >= min(rt)` wall — each of the 20 subjects gets an independent `t0` and only
        # one of them has to land high to ruin the chain.
        init_positions, init_exhausted = init_particles_from_prior(
            flat_space, num_chains, init_key, support=support, max_attempts=max_attempts,
        )
        if int(init_exhausted):
            logger.warning(
                "t0 rejection sampling exhausted for %d of %d warm-up starts; those fell "
                "back to clipping",
                int(init_exhausted), num_chains,
            )

        # One window adaptation per chain. target_acceptance_rate=0.8 avoids the degenerate
        # near-zero step sizes that 0.9 produces when the NLE posterior has high curvature.
        # Note the adaptation targets the lambda = 1 posterior but the step size is used from
        # lambda = 0 onward, where the target is the much broader prior.
        _, adapted_params = window_adaptation(
            blackjax.nuts,
            logdensity_fn,
            init_positions,
            smc_cfg["num_warmup"],
            warmup_key,
            num_chains=num_chains,
            target_acceptance_rate=0.8,
        )

        step_sizes = np.asarray(adapted_params["step_size"])
        logger.info("Adapted step sizes per chain: %s", step_sizes)

        # Drop, do not repair: see DEGENERATE_STEP_SIZE. Dropping here rather than after
        # sampling also saves running SMC on a chain whose result would be discarded.
        healthy, num_dropped = healthy_chains(step_sizes)
        if healthy.size == num_chains and np.all(step_sizes < DEGENERATE_STEP_SIZE):
            logger.error(
                "All %d chains produced a degenerate step size after warmup; keeping them "
                "all — these results are not usable",
                num_chains,
            )
        elif num_dropped:
            logger.warning(
                "Dropping %d of %d chains whose step size collapsed below %.1g (%s). A "
                "collapsed adaptation is reported, not repaired — a repaired chain stays "
                "under-dispersed and still breaks R-hat.",
                num_dropped, num_chains, DEGENERATE_STEP_SIZE,
                step_sizes[step_sizes < DEGENERATE_STEP_SIZE],
            )

        kept = jnp.asarray(healthy)
        mcmc_parameters = jax.tree.map(lambda leaf: leaf[kept], adapted_params)
        num_kept = int(kept.size)

        # One prior cloud per chain. Sharing a single cloud would leave any gap in that one
        # draw invisible to every between-chain comparison — and the between-chain spread is
        # the standard error on the log marginal likelihood.
        #
        # Pulled into the support for the same reason as the warm-up starts: a particle whose
        # `t0` is above some subject's fastest RT sits on the likelihood's flat floor and is
        # resampled away, so leaving them in just burns effective particles.
        initial_particles, cloud_exhausted = jax.vmap(
            lambda key: init_particles_from_prior(
                flat_space, num_particles, key, support=support, max_attempts=max_attempts,
            ),
        )(jax.random.split(cloud_key, num_kept))
        if int(jnp.sum(cloud_exhausted)):
            logger.warning(
                "t0 rejection sampling exhausted for %d of %d initial-cloud particles; those "
                "fell back to clipping",
                int(jnp.sum(cloud_exhausted)), num_kept * num_particles,
            )

        result = tempered_smc(
            chain_key,
            flat_space.log_prob,
            log_likelihood_fn,
            initial_particles,
            mcmc_parameters,
            num_integration_steps=smc_cfg["num_integration_steps"],
            target_ess=smc_cfg["target_ess"],
            num_mcmc_steps=smc_cfg["num_mcmc_steps"],
            map_chains=smc_cfg["map_chains"],
        )

        logger.info(
            "Weight ESS per chain (pre-resample): %s",
            np.round(np.asarray(result.weight_ess), 3),
        )
        logger.info(
            "Distinct particles per chain: %s of %d after SMC, %s after the final resample",
            np.asarray(result.num_unique_smc).tolist(), num_particles,
            np.asarray(result.num_unique).tolist(),
        )

        min_fraction = smc_cfg["min_unique_fraction"]
        if bool(np.any(np.asarray(result.num_unique) < min_fraction * num_particles)):
            # Which of the two counts is low says which stage to blame, so the warning reports
            # both rather than naming a single cause.
            logger.warning(
                "Particle degeneracy: only %s of %d stored draws per chain are distinct, "
                "below the %.0f%% floor, so their ESS and R-hat overstate what was sampled. "
                "Distinct before the final resample: %s. If that is low too, mutation is not "
                "rediversifying and hierarchical_recovery.smc.num_mcmc_steps should go up; if "
                "it is full, the loss is the final resample against non-uniform weights.",
                np.asarray(result.num_unique).tolist(), num_particles,
                100 * min_fraction, np.asarray(result.num_unique_smc).tolist(),
            )

        return result, initial_particles, num_dropped

    def write_population(label, result, initial_particles, num_dropped, data, context,
                         log_theta_true, prior_ds, pop_idx, path):
        """Assemble the DataTree for one fit, check convergence, and write it."""
        dt = _build_population_datatree(
            result=result,
            flat_space=flat_space,
            data=data,
            context=context,
            log_theta_true=log_theta_true,
            param_names=param_names,
            pop_idx=pop_idx,
            num_dropped=num_dropped,
        )
        dt["prior"] = prior_ds
        # The starting cloud is kept too, under its own name: it is a genuine diagnostic of
        # what SMC began from, but it is *not* the prior — it is support-restricted, and
        # tempered SMC never corrects its initial distribution by weighting. Each chain has
        # its own; pooled here.
        dt["smc_initial_particles"] = _particle_dataset(
            initial_particles.reshape(-1, initial_particles.shape[-1]), flat_space, param_names,
        )
        _log_convergence(dt, label, smc_cfg["max_rhat"], smc_cfg["min_ess"])
        logger.info(
            "%s log marginal likelihood: %.3f (per chain %s)",
            label,
            dt.attrs["log_marginal_likelihood_mean"],
            np.round(np.asarray(result.log_marginal_likelihood), 3).tolist(),
        )
        logger.info("Saving %s results to %s", label, path)
        dt.to_netcdf(path)

    test_key = jax.random.key(cfg["test_seed"])

    for pop_idx in range(num_populations):
        logger.info("Population %d / %d", pop_idx + 1, num_populations)

        test_key, data_key, sampling_key, prior_key = jax.random.split(test_key, 4)

        data, context = sampler(data_key, num_trials, num_subjects, **prior_cfg)
        logger.info("Simulated data shape: %s", data.shape)

        # All subjects have the same trial count, mask is all True
        mask = jnp.ones((num_subjects, num_trials), dtype=bool)

        min_rt = min_valid_rt(data[:, :, 0])
        log_theta_true = reconstruct_from_dict(context)

        # The `prior` group is the *model's* prior — untruncated, matching what generated the
        # data and what `flat_space.log_prob` evaluates. Drawn afresh rather than taken from
        # the SMC starting cloud, which is support-restricted and would halve the apparent
        # prior SD of `t0`: measured over 20 000 draws on the RDM config, the restricted
        # cloud's SD is 0.529x this one's for `mu[t0]` and within 1.2% for everything else.
        # Posterior contraction is reported as 1 - (sd_post/sd_prior)^2, which squares that,
        # so a genuine contraction to 0.75 would read as 0.107 against the restricted cloud.
        num_prior_draws = num_chains * num_particles
        prior_ds = _particle_dataset(
            flat_space.sample_particles(prior_key, num_prior_draws), flat_space, param_names,
        )

        logger.info("Running approximate recovery...")
        sampling_key, approx_key = jax.random.split(sampling_key)
        result, initial_particles, num_dropped = recover_population(
            approx_key, data, mask, likelihood_factory_approx, min_rt,
        )
        logger.info("SMC converged in %s iterations", np.asarray(result.num_iterations))
        write_population(
            "approx", result, initial_particles, num_dropped, data, context, log_theta_true,
            prior_ds, pop_idx, f"hierarchical_recovery_pop{pop_idx}_approx.nc",
        )

        if model_cfg.get("run_reference_recovery", False):
            logger.info("Running reference recovery...")
            sampling_key, ref_key = jax.random.split(sampling_key)
            ref_likelihood = instantiate(model_hier_cfg["likelihood_factory_ref"])
            result_ref, initial_ref, dropped_ref = recover_population(
                ref_key, data, mask, ref_likelihood, min_rt,
            )
            logger.info("Ref SMC converged in %s iterations", np.asarray(result_ref.num_iterations))
            write_population(
                "ref", result_ref, initial_ref, dropped_ref, data, context, log_theta_true,
                prior_ds, pop_idx, f"hierarchical_recovery_pop{pop_idx}_ref.nc",
            )


if __name__ == "__main__":
    main()
