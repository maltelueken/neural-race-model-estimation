"""Utility functions."""

import numpy as np
from scipy import stats


def constant_list(length, value):
    return (value,) * length


def get_decay_steps(num_epochs, num_batches):
    return num_epochs * num_batches


def sample_random_num_obs(batch_shape, rng):
    return dict(num_obs=rng.integers(100, 1000))


def truncated_normal_rvs(
    loc,
    scale,
    lower=0.0,
    size=1,
    random_state=None,
) -> np.ndarray:
    quantile_l = stats.norm.cdf(lower, loc=loc, scale=scale)

    if random_state is not None:
        probs = random_state.uniform(quantile_l, 1.0, size=size)
    else:
        probs = np.random.default_rng().uniform(quantile_l, 1.0, size=size)

    return stats.norm.ppf(
        probs,
        loc=loc,
        scale=scale,
    )