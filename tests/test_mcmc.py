"""Tests for the multi-chain MCMC helpers.

The point of `warmup_multiple_chains` is that R-hat becomes able to fail. These
tests pin that property down: chains must end up in genuinely different states,
and a sampler that misses half a bimodal posterior must be flagged.
"""

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from confrdm_jax.mcmc import inference_loop_multiple_chains, warmup_multiple_chains

NUM_CHAINS = 4
SEPARATION = 6.0


def _rhat(positions):
    """Split-free R-hat on the first coordinate; positions is (draws, chains, dim)."""
    x = np.asarray(positions[:, :, 0])
    n, m = x.shape
    chain_means, chain_vars = x.mean(axis=0), x.var(axis=0, ddof=1)
    between = n * chain_means.var(ddof=1)
    within = chain_vars.mean()
    return float(np.sqrt(((n - 1) / n * within + between / n) / within))


def _bimodal_logdensity(x):
    """Two well-separated Gaussians; NUTS cannot cross between them."""
    return jnp.logaddexp(
        -0.5 * jnp.sum((x - SEPARATION) ** 2), -0.5 * jnp.sum((x + SEPARATION) ** 2),
    )


def _unimodal_logdensity(x):
    return -0.5 * jnp.sum(x**2)


def _sample(logdensity_fun, init_positions, num_warmup=400, num_samples=1000, seed=0):
    warmup_key, sampling_key = jax.random.split(jax.random.PRNGKey(seed))
    states, params = warmup_multiple_chains(
        blackjax.nuts, logdensity_fun, init_positions, num_warmup, warmup_key,
    )
    nuts_kernel = blackjax.nuts.build_kernel()

    def kernel(key, state, chain_params):
        return nuts_kernel(key, state, logdensity_fun, **chain_params)

    positions, info = inference_loop_multiple_chains(
        sampling_key, kernel, states, num_samples, NUM_CHAINS, params,
    )
    return positions, info, states, params


class TestWarmupMultipleChains:
    def test_chains_are_tuned_independently(self):
        """Each chain gets its own step size and mass matrix, not a shared one."""
        init = jax.random.normal(jax.random.PRNGKey(1), (NUM_CHAINS, 2)) * 3.0
        _, _, states, params = _sample(_unimodal_logdensity, init)

        assert params["step_size"].shape[0] == NUM_CHAINS
        assert params["inverse_mass_matrix"].shape[0] == NUM_CHAINS
        # Independent adaptation means the tuning genuinely differs.
        assert len(np.unique(np.asarray(params["step_size"]))) > 1

    def test_chains_end_warmup_in_different_states(self):
        """The whole point: no two chains start sampling from the same position."""
        init = jax.random.normal(jax.random.PRNGKey(2), (NUM_CHAINS, 2)) * 3.0
        _, _, states, _ = _sample(_unimodal_logdensity, init)

        positions = np.asarray(states.position)
        assert positions.shape[0] == NUM_CHAINS
        assert np.std(positions, axis=0).min() > 0.0

    def test_shared_init_hides_a_missed_mode(self):
        """Baseline: replicating one state gives R-hat ~ 1 on a half-missed posterior."""
        start = jnp.full((NUM_CHAINS, 2), SEPARATION)
        positions, _, _, _ = _sample(_bimodal_logdensity, start)

        # Every chain is stuck in the mode it started in.
        assert np.mean(np.asarray(positions[:, :, 0]) > 0) == pytest.approx(1.0)
        # ...and R-hat does not notice.
        assert _rhat(positions) < 1.01

    def test_overdispersed_init_detects_a_missed_mode(self):
        """The property the fix buys: R-hat can now fail."""
        start = jnp.array(
            [[SEPARATION] * 2, [SEPARATION] * 2, [-SEPARATION] * 2, [-SEPARATION] * 2],
        )
        positions, _, _, _ = _sample(_bimodal_logdensity, start)

        assert _rhat(positions) > 1.1

    def test_unimodal_target_still_converges(self):
        """Dispersed starts must not manufacture false alarms on an easy target."""
        init = jax.random.normal(jax.random.PRNGKey(3), (NUM_CHAINS, 2)) * 3.0
        positions, info, _, _ = _sample(_unimodal_logdensity, init)

        assert _rhat(positions) < 1.01
        assert int(np.asarray(info.is_divergent).sum()) == 0
        assert float(jnp.mean(positions)) == pytest.approx(0.0, abs=0.1)


class TestInferenceLoopMultipleChains:
    def test_accepts_a_prebound_kernel(self):
        """The kernel_params=None path stays available for a single shared tuning."""
        nuts = blackjax.nuts(
            _unimodal_logdensity, step_size=0.5, inverse_mass_matrix=jnp.ones(2),
        )
        states = jax.vmap(nuts.init)(jnp.zeros((NUM_CHAINS, 2)))
        positions, _ = inference_loop_multiple_chains(
            jax.random.PRNGKey(0), nuts.step, states, 200, NUM_CHAINS,
        )
        assert positions.shape == (200, NUM_CHAINS, 2)
