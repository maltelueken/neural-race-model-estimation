"""Adapters."""

import bayesflow as bf

def create_nle_adapter_two_accumulators(param_names):
    return (
        bf.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .broadcast(param_names, to="x", expand=1)
        .drop("num_obs")
        .as_set(["x"])
        .log(param_names)
        .concatenate(param_names, into="inference_conditions")
        .rename("x", "inference_variables")
    )


def create_nle_adapter_single_accumulator(param_names):
    return (
        bf.Adapter()
        .to_array()
        # .convert_dtype("float64", "float32")
        .drop("num_obs")
        .expand_dims("x", axis=-1)
        .broadcast(param_names, to="x", expand=(1,))
        .as_set(["x"])
        .log(param_names)
        .concatenate(param_names, into="inference_conditions")
        .rename("x", "inference_variables")
    )


def create_npe_adapter_two_accumulators(param_names):
    return (
        bf.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .broadcast("num_obs", to="x", exclude=(-2, -1), squeeze=-1)
        .as_set(["x"])
        .log(param_names)
        .sqrt("num_obs")
        .concatenate(param_names, into="inference_variables")
        .concatenate(["x"], into="summary_variables")
        .rename("num_obs", "inference_conditions")
        .keep(
            ["inference_variables", "inference_conditions", "summary_variables"]
        )
    )