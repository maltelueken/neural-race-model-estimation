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
    # Generate noise for all accumulators and time steps at once
    noise = jax.random.normal(key, shape=mu.shape)

    # 1. FIX: Scale drift by dt (Euler-Maruyama integration)
    # dx = v * dt + s * dW
    increments = (mu * dt) + (s * noise * jnp.sqrt(dt))

    # Calculate trajectories
    x = jnp.cumsum(increments, axis=-1)

    # Determine crossings
    crossing = x >= b

    # Find the index of the first crossing
    # argmax returns the first index where True occurs; if all False, returns 0
    first_idx = jnp.argmax(crossing, axis=-1)

    # Calculate the time of crossing
    # We use (idx + 1) because cumsum at index 0 is actually step 1
    first_time = (first_idx + 1) * dt

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
