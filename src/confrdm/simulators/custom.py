


import bayesflow as bf
import numpy as np

from bayesflow.utils.decorators import allow_batch_size


class CustomSimulator(bf.simulators.Simulator):
    def __init__(
        self,
        prior_simulator,
        design_simulator,
        experiment_simulator,
    ):
        self.prior_simulator = prior_simulator
        self.design_simulator = design_simulator
        self.experiment_simulator = experiment_simulator

    @allow_batch_size
    def sample(self, batch_shape, **kwargs):
        prior_dict = self.prior_simulator.sample(batch_shape)

        design_dict = self.design_simulator.sample(batch_shape)

        design_dict.update(**kwargs)

        sims_dict = self.experiment_simulator.sample(
            batch_shape, **prior_dict, **design_dict
        )

        data = prior_dict | design_dict | sims_dict

        data = {
            key: np.expand_dims(value, axis=-1) if np.ndim(value) == 1 else value
            for key, value in data.items()
        }

        return data