
import bayesflow as bf
import keras

from jax.scipy import stats

class LikelihoodApproximator:
    def __init__(self, approximator):
        self.approximator = approximator

    def _prepare_data_adapted(
        self, data, ldj_inference, log_det_jac
    ):
        # Standardize conditions
        for key in self.approximator.CONDITION_KEYS:
            if key in self.approximator.standardize and key in data:
                data[key] = self.approximator.standardize_layers[key](data[key])

        # Standardize inference variables
        if "inference_variables" in data and "inference_variables" in self.approximator.standardize:
            result = self.approximator.standardize_layers["inference_variables"](
                data["inference_variables"], log_det_jac=log_det_jac
            )
            if log_det_jac:
                data["inference_variables"], ldj_std = result
                ldj_inference += ldj_std
            else:
                data["inference_variables"] = result

        # Convert all data to tensors
        data = keras.tree.map_structure(keras.ops.convert_to_tensor, data)

        if log_det_jac:
            return data, ldj_inference
        return data
    
    def _prepare_data(self, data, log_det_jac, **kwargs):
        adapted = self.approximator.adapter(data, strict=False, log_det_jac=log_det_jac, **kwargs)

        if log_det_jac:
            data, ldj = adapted
            ldj_inference = ldj.get("inference_variables", 0.0)
        else:
            data = adapted
            ldj_inference = None

        return self._prepare_data_adapted(data, ldj_inference, log_det_jac)
    
    def _log_pdf_sf(self, data, log_det_jac, **kwargs):
        z, log_pdf = self.approximator.inference_network(
            data["inference_variables"],
            conditions=data["inference_conditions"],
            inverse=False,
            density=True,
            **bf.utils.filter_kwargs(kwargs, self.approximator.inference_network.log_prob),
        )

        # Change of variables formula.
        log_pdf += log_det_jac

        log_sf = stats.norm.logsf(z)

        return log_pdf, log_sf
    
    def log_pdf_sf(self, data, **kwargs):
        # Adapt, optionally standardize and convert to tensor. Keep track of log_det_jac.
        data, log_det_jac = self._prepare_data(data, log_det_jac=True, **kwargs)

        return self._log_pdf_sf(data, log_det_jac)
    