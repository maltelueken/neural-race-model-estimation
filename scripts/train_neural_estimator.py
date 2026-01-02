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

from confrdm.simulators.rdmc import sample_rdmc_prior_single_accumulator, simulate_rdmc_single_accumulator
from confrdm.utils import sample_random_num_obs

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    simulator = instantiate(cfg["simulator"], _convert_="partial")
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
