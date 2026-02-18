import jax
import jax.numpy as jnp

def smc_inference_loop(rng_key, smc_kernel, initial_state, max_steps=100):
    """Run the tempered SMC algorithm and retain all intermediate states.

    Returns (n_iter, final_state, history) where `history` is a PyTree
    with arrays shaped (max_steps+1, ...) storing the state at each iteration index.
    """
    def cond(carry):
        i, state, _k = carry
        return state.lmbda < 1

    @jax.jit
    def one_step(carry):
        i, state, k = carry
        k, subk = jax.random.split(k, 2)
        next_state, _ = smc_kernel(subk, state)
        return i + 1, next_state, k

    carry0 = (0, initial_state, rng_key)
    n_iter, final_state, _k = jax.lax.while_loop(cond, one_step, carry0)

    return n_iter, final_state
