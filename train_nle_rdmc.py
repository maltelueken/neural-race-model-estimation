import os

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"

import logging

import bayesflow as bf
import hydra
import matplotlib.pyplot as plt
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from rdmc import rdmc_prior, rdmc_experiment_simple

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    prior_args = instantiate(cfg["simulator"]["prior_simulator"])
    experiment_args = instantiate(cfg["simulator"]["experiment_simulator"])

    def prior_fun():
        return rdmc_prior(**prior_args)

    def experiment_fun(**kwargs):
        return rdmc_experiment_simple(**kwargs, **experiment_args)

    def random_num_obs(batch_shape):
        return dict(num_obs=np.random.default_rng(cfg["seed"]).integers(100, 1000))

    simulator = bf.simulators.make_simulator([prior_fun, experiment_fun], meta_fn=random_num_obs)

    approximator = instantiate(cfg["approximator"], _convert_="partial")
    optimizer = instantiate(cfg["optimizer"], _convert_="partial")

    approximator.compile(optimizer)

    callbacks = instantiate(cfg["callbacks"], _convert_="partial")

    history = approximator.fit(
        epochs=cfg["epochs"],
        num_batches=cfg["iterations_per_epoch"],
        batch_size=cfg["batch_size"],
        callbacks=callbacks,
        simulator=simulator
    )

    _ = bf.diagnostics.plots.loss(history)

    plt.savefig("loss_history.png")


if __name__ == "__main__":
    main()
