"""Shared building blocks for the conflict racing diffusion model.

Two things live here: the gamma-shaped conflict pulse, and the
Euler-Maruyama trial simulator that every CRDM sampler is built on.

The conflict signal is a normalised gamma function of time.  Because the
accumulators integrate their drift, it is convenient to have both the pulse
itself and its derivative: the *derivative* is the instantaneous drift added to
an accumulator, and the *pulse* is the corresponding integrated drift, which is
what the Volterra reference solver needs.  Normalising by the mode
(``exp(1) * x / ((a_shape - 1) * tau)``) makes ``amp`` the peak height of the
integrated signal regardless of ``tau``, so amplitude and time scale are not
confounded in the prior.
"""

from typing import Tuple

import jax
import jax.numpy as jnp


def normalized_gamma(x, amp, tau, a_shape):
    """Gamma-shaped conflict pulse, scaled so its peak value is `amp`.

    This is the *integrated* conflict drift M_a(t): the contribution of the
    conflict signal to an accumulator's position at time `x`.

    Args:
        x: Time, strictly positive.
        amp: Peak height. May be negative, which flips the pulse's sign; CRDM
            samplers use that to route the pulse to the other accumulator and
            pass ``jnp.abs(amp)`` here.
        tau: Time scale; the peak falls at ``(a_shape - 1) * tau``.
        a_shape: Gamma shape parameter. All callers use 2.0.
    """
    return (amp
        * jnp.exp(-x / tau)
        * (jnp.exp(1) * x / (a_shape - 1) / tau) ** (a_shape - 1)
    )


def normalized_gamma_derivative(x, amp, tau, a_shape):
    """Time derivative of :func:`normalized_gamma` — the instantaneous drift.

    This is what gets added to an accumulator's drift rate at each time step.
    It is positive before the pulse peak and negative after, so a congruent
    pulse first accelerates the accumulator and then decelerates it back.

    Undefined at ``x = 0`` as written (the ``(a_shape - 1) / x`` term), though
    the limit is finite for ``a_shape = 2``; every caller evaluates on a grid
    starting at ``dt > 0``.
    """
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
    """Race one trial of accumulators with time-varying drift to a fixed bound.

    Integrates ``dx = mu(t) dt + s dW`` on a fixed grid for every accumulator
    in parallel, tests for boundary crossings (grid plus Brownian bridge, see
    below), and returns the winner of the race.

    Args:
        mu: Instantaneous drift per accumulator per time step, shape
            ``(num_accumulators, num_steps)``. The number of columns *is* the
            time grid: step ``i`` covers ``(i * dt, (i + 1) * dt]``, so the
            integration horizon is ``num_steps * dt``.
        b: Absorbing boundary, shared by all accumulators.
        s: Diffusion coefficient, shape ``(num_accumulators, 1)`` so it
            broadcasts across time.
        t0: Non-decision time, added to the decision time. Training simulators
            pass 0 so the flow models decision times only.
        dt: Time step.
        key: PRNG key.

    Returns:
        ``(rt, resp)``. If some accumulator crossed, `rt` is its crossing time
        plus `t0` and `resp` is its index. If none did within the horizon, the
        **censoring sentinel** ``(-1.0, -1)`` is returned: that encodes the
        observation ``T > t_max``, not missing data, and every consumer of
        simulator output has to handle it. ``flows.loss_fn`` scores it by
        ``log S(t_max)``; ``scripts/parameter_recovery.py`` excludes it from
        the ``t0`` initialisation.
    """
    key_noise, key_bridge, key_time = jax.random.split(key, 3)

    # Generate noise for all accumulators and time steps at once
    noise = jax.random.normal(key_noise, shape=mu.shape)

    # Euler-Maruyama increment: dx = mu(t) dt + s dW,  dW ~ N(0, dt)
    increments = (mu * dt) + (s * noise * jnp.sqrt(dt))

    # Trajectories, all accumulators at once.  Every path starts at x = 0.
    x = jnp.cumsum(increments, axis=-1)

    # Crossings visible on the grid itself
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
    # ~3x at dt = 0.005 and ~5-20x at dt = 0.0005, for ~1.55x the runtime, and
    # improves the scaling in dt from dt^0.66 to dt^0.9.  (Concretely, at
    # v_c = 2.5, tau = 0.10, b = 1.0: 32.3 -> 10.3 ms of lateness at
    # dt = 0.005, and 7.7 -> 2.0 ms at dt = 0.0005.)  Shrinking dt tenfold
    # instead costs 9x and only buys dt^0.66, so the bridge is much the better
    # trade.
    #
    # What it does not fix: integrating a sharp gamma pulse as `mu * dt` is
    # O(dt) accurate however crossings are detected, so a residual of ~2 ms
    # survives at tau = 0.05 and responds only to smaller dt.
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

    # `argmax` returns 0 when nothing crossed, so a separate check is needed to
    # tell "crossed at the first step" from "never crossed".
    did_cross = jnp.any(crossing, axis=-1)

    # Non-crossers get an infinite first-passage time so they lose the race
    # rather than winning it at t = 0.
    fpt = jnp.where(did_cross, first_time, jnp.inf)

    # Race: the accumulator with the smallest first-passage time wins.
    min_fpt = jnp.min(fpt)
    winner_idx = jnp.argmin(fpt)

    # If no accumulator crossed within the horizon, emit the censoring
    # sentinel (-1.0, -1) rather than a time — see the docstring.
    any_one_crossed = jnp.any(did_cross)

    rt = jnp.where(any_one_crossed, min_fpt + t0, -1.0)
    resp = jnp.where(any_one_crossed, winner_idx, -1)

    return rt, resp
