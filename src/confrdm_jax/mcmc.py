import blackjax
import jax


def inference_loop_multiple_chains(
    rng_key, kernel, initial_state, num_samples, num_chains,
):
    @jax.jit
    def one_step(states, rng_key):
        keys = jax.random.split(rng_key, num_chains)
        states, infos = jax.vmap(kernel)(keys, states)
        return states, (states.position, infos)

    keys = jax.random.split(rng_key, num_samples)
    _, (positions, infos) = jax.lax.scan(one_step, initial_state, keys)

    return positions, infos


def warmup(sampler_fun, logdensity_fun, init_position, num_steps, rng_key, **kwargs):
    adapt = blackjax.window_adaptation(sampler_fun, logdensity_fun, **kwargs)
    (last_state, parameters), _ = adapt.run(rng_key, init_position, num_steps=num_steps)
    kernel = sampler_fun(logdensity_fun, **parameters).step
    return kernel, last_state, parameters
