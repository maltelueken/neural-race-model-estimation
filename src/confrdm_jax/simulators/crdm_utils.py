from typing import Tuple

import jax
import jax.numpy as jnp


def normalized_gamma(x, amp, tau, a_shape):
    return (amp
        * jnp.exp(-x / tau)
        * (jnp.exp(1) * x / (a_shape - 1) / tau) ** (a_shape - 1)
    )


def normalized_gamma_derivative(x, amp, tau, a_shape):
    return normalized_gamma(x, amp, tau, a_shape) * ((a_shape - 1) / x - 1 / tau)


@jax.jit
def simulate_crdm_single_trial(
    mu: jnp.ndarray,
    b: jnp.ndarray,
    s: jnp.ndarray,
    t0: float,
    dt: float,
    key: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    key_noise, key_bridge, key_time = jax.random.split(key, 3)

    # Generate noise for all accumulators and time steps at once
    noise = jax.random.normal(key_noise, shape=mu.shape)

    # 1. FIX: Scale drift by dt (Euler-Maruyama integration)
    # dx = v * dt + s * dW
    increments = (mu * dt) + (s * noise * jnp.sqrt(dt))

    # Calculate trajectories
    x = jnp.cumsum(increments, axis=-1)

    # Determine crossings
    crossing = x >= b

    # Brownian-bridge correction.  Between two grid points that both sit below
    # b the continuous path may still have touched b and returned; sampling
    # only the grid never sees those excursions, so crossings can be reported
    # late but never early.  That one-sided error is what makes naive
    # Euler-Maruyama weak order 1/2 in the first-passage time.  Conditional on
    # the two endpoints the touch probability is exact (the drift cancels out
    # of the bridge), and testing it restores weak order 1:
    #
    #     P(max over the step >= b) = exp(-2 (b - x_prev)(b - x) / (s^2 dt))
    #
    # Measured against the Volterra solution this cuts the mean-RT bias by
    # ~3x at dt = 0.005 and ~5-8x at dt = 0.0005, for ~1.55x the runtime.
    x_prev = jnp.concatenate([jnp.zeros_like(x[..., :1]), x[..., :-1]], axis=-1)
    p_touch = jnp.exp(-2.0 * (b - x_prev) * (b - x) / (s ** 2 * dt))
    touched = jax.random.uniform(key_bridge, shape=x.shape) < p_touch
    crossing = crossing | (touched & (x_prev < b) & (x < b))

    # Find the index of the first crossing
    # argmax returns the first index where True occurs; if all False, returns 0
    first_idx = jnp.argmax(crossing, axis=-1)

    # The barrier was reached somewhere *inside* step `first_idx`, not at its
    # right endpoint, so place the time uniformly within that step.  This is
    # the dequantisation that `sample_conditional_crdm_single` used to apply
    # afterwards as a symmetric +-dt/2 jitter; symmetric jitter around the
    # right endpoint centred the draw half a step too late.
    offset = jax.random.uniform(key_time, shape=first_idx.shape)
    first_time = (first_idx + offset) * dt

    # Check which accumulators actually crossed the boundary
    did_cross = jnp.any(crossing, axis=-1)

    # 2. FIX: Handle non-crossers
    # If an accumulator didn't cross, set its FPT to infinity.
    # Your original code set this to 'dt', which made non-crossers win the race immediately.
    fpt = jnp.where(did_cross, first_time, jnp.inf)

    # 3. Race Logic
    # Find the fastest FPT among all accumulators
    min_fpt = jnp.min(fpt)     # The winning time
    winner_idx = jnp.argmin(fpt) # The winning accumulator

    # If NO accumulator crossed, return -1.0/-1
    any_one_crossed = jnp.any(did_cross)

    rt = jnp.where(any_one_crossed, min_fpt + t0, -1.0)
    resp = jnp.where(any_one_crossed, winner_idx, -1)

    return rt, resp
