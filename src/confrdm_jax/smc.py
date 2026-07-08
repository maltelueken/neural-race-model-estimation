import jax
import jax.numpy as jnp

def smc_inference_loop(rng_key, smc_kernel, initial_state, max_steps=200):
    """Run the tempered SMC algorithm until lmbda reaches 1.

    Args:
        rng_key: JAX PRNG key.
        smc_kernel: SMC step function (from blackjax.adaptive_tempered_smc.step).
        initial_state: Initial SMC state.
        max_steps: Safety cap on the number of iterations. Prevents infinite loops
            if lmbda fails to converge. Default 200.

    Returns:
        Tuple of (n_iter, final_state).
    """
    def cond(carry):
        i, state, _k = carry
        return (state.lmbda < 1) & (i < max_steps)

    @jax.jit
    def one_step(carry):
        i, state, k = carry
        k, subk = jax.random.split(k, 2)
        next_state, _ = smc_kernel(subk, state)
        return i + 1, next_state, k

    carry0 = (0, initial_state, rng_key)
    n_iter, final_state, _k = jax.lax.while_loop(cond, one_step, carry0)

    return n_iter, final_state
