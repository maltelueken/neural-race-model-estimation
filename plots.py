"""Plotting helper functions.
"""

import jax
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns

def create_delta_plot_rdmc(data):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    rt = data[:, 0]
    resp = data[:, 1]
    cond = data[:, 2]

    qs = np.linspace(0, 1, 19)[1:-1]

    mean_quantiles = np.array(
        [
            stats.mstats.mquantiles(
                rt[np.bitwise_and(resp == 1, cond == i)],
                qs,
                alphap=0.5,
                betap=0.5,
            )
            for i in (0, 1)
        ]
    )

    diff_quantiles = mean_quantiles[0, :] - mean_quantiles[1, :]

    ax.plot(mean_quantiles.mean(axis=0), diff_quantiles, "--o", color="black")

    ax.set_ylabel("Difference mean RT congruent vs. incongruent (ms)")
    ax.set_xlabel("Mean RT quantiles (ms)")
    
    fig.tight_layout()
    return fig, ax


def create_pushforward_plot_rdmc(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 5))
    rt = data[:, 0]
    resp = data[:, 1]
    cond = data[:, 2]

    for i, ax in enumerate((ax1, ax2)):
        sns.histplot(
            rt[np.bitwise_and(resp == 1, cond == i)], color="darkgreen", alpha=0.5, ax=ax
        )
        sns.histplot(
            rt[np.bitwise_and(resp == 0, cond == i)], color="maroon", alpha=0.5, ax=ax
        )
        sns.despine(ax=ax)
        ax.vlines([rt[np.bitwise_and(resp == 1, cond == i)].mean(), rt[np.bitwise_and(resp == 0, cond == i)].mean()], ymin = 0, ymax = 0.5, color = ["darkgreen", "maroon"], linestyle = '-', 
            transform=ax.get_xaxis_transform())

        ax.text(
            0.9,
            0.9,
            f"Acc: {str(np.round(resp[cond == i].mean(), 2))}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )

        ax.spines['bottom'].set_position('zero')

        ax.set_ylabel("")
        ax.set_yticks([])
        ax.set_xlabel("Simulated RTs (milliseconds)") 

    ax1.set_title("Incongruent")
    ax2.set_title("Congruent")

    fig.tight_layout()

    return fig, (ax1, ax2)


def create_profile_likelihood_plot(ll_fun, prior_draws, param_names):
    p_range = 0.9
    num_points = 100

    p_grid = np.linspace(prior_draws - p_range * prior_draws / 2, prior_draws + p_range * prior_draws / 2, num_points)

    x = np.tile(prior_draws, (num_points*len(prior_draws), 1))

    for i in range(len(prior_draws)):
        x[(i*num_points):((i+1)*num_points),i] = p_grid[:, i]
    # ll_prior = nle_logdensity_fun(prior_draws)

    log_prob = jax.vmap(ll_fun)(x)
    # log_prob = np.array([nle_logdensity_fun(e) for e in x])

    fig, axes = plt.subplots(1, len(param_names), figsize=(15, 3))

    for i, ax in enumerate(axes):
        ax.plot(x[(i*num_points):((i+1)*num_points), i], log_prob[(i*num_points):((i+1)*num_points)])
        ax.plot(prior_draws[i], ll_fun(prior_draws), "o", color="red")
        if param_names is not None:
            ax.set_xlabel(param_names[i])
        ax.set_ylabel("Log-likelihood")

    fig.tight_layout()

    return fig, axes