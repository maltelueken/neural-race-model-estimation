import jax
import jax.numpy as jnp

def smc_inference_loop(rng_key, smc_kernel, initial_state, max_steps=100):
    """Run the tempered SMC algorithm and retain all intermediate states.

    Returns (n_iter, final_state, history) where `history` is a PyTree
    with arrays shaped (max_steps+1, ...) storing the state at each iteration index.
    """
    def cond(carry):
        i, state, _k, _hist = carry
        return state.lmbda < 1

    # initialize history PyTree with an extra leading axis for time
    def init_history(state):
        def make_buffer(x):
            shp = (max_steps + 1,) + tuple(jnp.shape(x))
            return jnp.zeros(shp, dtype=jnp.asarray(x).dtype)
        return jax.tree.map(make_buffer, state)

    history0 = init_history(initial_state)
    history0 = jax.tree.map(lambda h, x: h.at[0].set(x), history0, initial_state)

    @jax.jit
    def one_step(carry):
        i, state, k, history = carry
        k, subk = jax.random.split(k, 2)
        next_state, _ = smc_kernel(subk, state)
        # store the new state at index i+1
        history = jax.tree.map(lambda h, x: h.at[i + 1].set(x), history, next_state)
        return i + 1, next_state, k, history

    carry0 = (0, initial_state, rng_key, history0)
    n_iter, final_state, _k, final_history = jax.lax.while_loop(cond, one_step, carry0)

    return n_iter, final_state, final_history
