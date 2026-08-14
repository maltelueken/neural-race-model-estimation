"""BlackJAX NUTS driving helpers for the recovery scripts.

Two pieces: window adaptation (:func:`warmup`, :func:`warmup_multiple_chains`)
and the sampling loop (:func:`inference_loop_multiple_chains`).  They are split
so that chains can be adapted *independently* and then advanced in lockstep,
each carrying its own step size and mass matrix.  ``warmup_multiple_chains`` is
used by the hierarchical SMC path too, for the mutation kernel's tuning.

Everything here operates on unconstrained positions.  The models parameterise
in log space, so the log-density handed in is expected to include the
log-transform Jacobian already (see ``log_prior`` in
``scripts/parameter_recovery.py``).
"""

import blackjax
import jax
from blackjax.adaptation.base import get_filter_adapt_info_fn


def inference_loop_multiple_chains(
    rng_key, kernel, initial_state, num_samples, num_chains, kernel_params=None,
):
    """Run `num_chains` chains in lockstep, one vmapped kernel call per step.

    Args:
        rng_key: PRNG key; split per step and then per chain.
        kernel: called as ``kernel(key, state)`` when `kernel_params` is None —
            i.e. the tuning constants are already bound — otherwise as
            ``kernel(key, state, params)``.
        kernel_params: optional pytree whose leaves carry a leading axis of
            length `num_chains`, so each chain uses the step size and mass
            matrix that its *own* window adaptation produced. Required when the
            chains were warmed up independently, since their tuning differs.
        initial_state: per-chain sampler states, leading axis `num_chains`.
        num_samples: draws to record per chain; nothing is discarded.
        num_chains: number of chains advanced in lockstep.
    """
    @jax.jit
    def one_step(states, rng_key):
        keys = jax.random.split(rng_key, num_chains)
        if kernel_params is None:
            states, infos = jax.vmap(kernel)(keys, states)
        else:
            states, infos = jax.vmap(kernel)(keys, states, kernel_params)
        return states, (states.position, infos)

    keys = jax.random.split(rng_key, num_samples)
    _, (positions, infos) = jax.lax.scan(one_step, initial_state, keys)

    return positions, infos


def warmup(sampler_fun, logdensity_fun, init_position, num_steps, rng_key, **kwargs):
    """Run window adaptation for a single chain.

    Args:
        sampler_fun: BlackJAX sampler constructor, e.g. ``blackjax.nuts``.
        logdensity_fun: Unnormalised log posterior on unconstrained positions.
        init_position: Starting position pytree (no chain axis).
        num_steps: Adaptation steps.
        rng_key: PRNG key.
        **kwargs: Forwarded to ``blackjax.window_adaptation``.

    Returns:
        ``(kernel, last_state, parameters)`` with the tuning constants already
        bound into `kernel`, so it is called as ``kernel(key, state)``.  Use
        :func:`warmup_multiple_chains` instead when running several chains —
        replicating one warmed-up state across chains makes R-hat vacuous.
    """
    adapt = blackjax.window_adaptation(sampler_fun, logdensity_fun, **kwargs)
    (last_state, parameters), _ = adapt.run(rng_key, init_position, num_steps=num_steps)
    kernel = sampler_fun(logdensity_fun, **parameters).step
    return kernel, last_state, parameters


def warmup_multiple_chains(
    sampler_fun, logdensity_fun, init_positions, num_steps, rng_key, **kwargs,
):
    """Run window adaptation independently for each chain.

    Replicating a single warmed-up state across chains — which is what
    ``jax.vmap(lambda _: last_state)(...)`` does — leaves R-hat with nothing to
    measure but Monte-Carlo noise, because the between-chain variance starts at
    zero and the chains begin inside whichever mode the one warm-up found.

    That failure is not hypothetical.  On a controlled bimodal target with
    modes at +-6 (4 chains x 2000 draws), the shared-state init reports max
    R-hat = 1.001 while putting 100% of its mass in a single mode; an
    overdispersed init reports R-hat = 1.734 and splits 50/50, which is the
    truth.  The draws from the shared init are not *biased* — every chain gets
    a fresh key per step, and they start from a state adaptation has already
    brought to stationarity — but the diagnostic reported alongside them cannot
    fail, so it carries no information.

    Adapting per chain gives starts that are both overdispersed *and*
    converged, so R-hat measures what Gelman-Rubin assumes it measures. The
    cost is `num_chains` times the warm-up work.

    Adaptation info is filtered out: keeping it for every chain and step is
    pure memory with no downstream consumer.

    Args:
        sampler_fun: BlackJAX sampler constructor, e.g. `blackjax.nuts`.
        logdensity_fun: shared unnormalised log posterior.
        init_positions: pytree of starting positions with a leading axis of
            length `num_chains`. These should be genuinely dispersed and must
            all lie inside the support — see the `init` block of
            `conf_jax/config.yaml` and `_sample_init_positions_in_support` in
            `scripts/parameter_recovery.py` for the `t0 < min(rt)` constraint.
        num_steps: adaptation steps per chain.
        rng_key: PRNG key; split once per chain.
        **kwargs: forwarded to `blackjax.window_adaptation`.

    Returns:
        `(last_states, parameters)`, both with a leading chain axis. Pass
        `parameters` straight to `inference_loop_multiple_chains` as
        `kernel_params`.
    """
    num_chains = jax.tree.leaves(init_positions)[0].shape[0]

    adapt = blackjax.window_adaptation(
        sampler_fun,
        logdensity_fun,
        adaptation_info_fn=get_filter_adapt_info_fn(),
        **kwargs,
    )

    def run_one(key, position):
        (last_state, parameters), _ = adapt.run(key, position, num_steps=num_steps)
        return last_state, parameters

    return jax.vmap(run_one)(jax.random.split(rng_key, num_chains), init_positions)
