"""Classifier two-sample test (C2ST) between neural and analytic recovery.

Compares the posterior samples produced by the neural likelihood estimator
("approx") against those from the analytic / reference likelihood ("ref") for a
parameter-recovery run, using
``bayesflow.diagnostics.metrics.classifier_two_sample_test``.

A C2ST trains a classifier to tell the two sets of posterior samples apart.  An
accuracy near 0.5 means the neural and analytic posteriors are indistinguishable
(the neural estimator recovers the same posterior); an accuracy near 1.0 means
they differ.

Two cases, matching the two recovery scripts:

- ``single``: each recovered dataset is an independent single-subject fit
  (``parameter_recovery.py``).  One C2ST is run **per subject**, comparing the
  joint posterior over all parameters for that subject.  This is repeated
  separately for each number of trials (one recovery directory per trial count,
  listed in ``c2st.single_recovery_dirs``).
- ``hierarchical``: one joint model is fit to all subjects at once
  (``parameter_recovery_hierarchical.py``).  One C2ST is run for the
  **entire model**, per population file, comparing the joint posterior over
  population-level (``mu``, ``sigma``) and all subject-level parameters.

Configuration lives under the ``c2st`` group in ``conf_jax/config.yaml``.

Examples
--------
Single-subject (one C2ST per subject, separately per number of trials)::

    python scripts/c2st_recovery.py c2st.mode=single \
        "c2st.single_recovery_dirs=['multirun/rdm/.../test_num_obs=50/train_steps=500000', \
                                    'multirun/rdm/.../test_num_obs=500/train_steps=500000']"

Hierarchical (one C2ST per population for the whole model)::

    python scripts/c2st_recovery.py c2st.mode=hierarchical \
        c2st.recovery_dir=outputs/rdm/.../train_steps=500000
"""

import csv
import logging
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from xarray import open_datatree

from bayesflow.diagnostics.metrics import classifier_two_sample_test

logger = logging.getLogger(__name__)


def _pooled_param_names(dt):
    """Return the list of parameter names stored on the DataTree."""
    param_names = dt.attrs.get("param_names")
    if param_names is None:
        raise ValueError("DataTree has no 'param_names' attribute.")
    return list(param_names)


def _subject_features(posterior, param_names, subject):
    """Stack one subject's posterior into a ``(num_samples, num_params)`` array.

    Chains and draws are pooled into a single sample dimension; the parameters
    become the feature columns.
    """
    cols = []
    for name in param_names:
        da = posterior[name].sel(subject=subject).stack(sample=("chain", "draw"))
        cols.append(np.asarray(da.transpose("sample").values))
    return np.stack(cols, axis=-1)


def _model_features(posterior, param_names):
    """Stack the full hierarchical posterior into ``(num_samples, num_features)``.

    Features are, in order: population mean ``mu`` (one column per parameter),
    population SD ``sigma`` (one per parameter), and every subject-level
    parameter (one column per subject per parameter).  Chains and draws are
    pooled into the sample dimension.
    """
    cols = []
    for pop_var in ("mu", "sigma"):
        if pop_var in posterior:
            da = posterior[pop_var].stack(sample=("chain", "draw"))
            cols.append(np.asarray(da.transpose("sample", "param").values))
    for name in param_names:
        da = posterior[name].stack(sample=("chain", "draw"))
        cols.append(np.asarray(da.transpose("sample", "subject").values))
    return np.concatenate(cols, axis=-1)


def run_single(approx_path, ref_path, c2st_kwargs):
    """Run one C2ST per subject; return ``(num_obs, rows)``.

    ``num_obs`` is the number of trials per subject, read from the recovery
    file. ``rows`` is a list of ``(subject, accuracy)``.
    """
    approx_dt = open_datatree(approx_path)
    ref_dt = open_datatree(ref_path)

    param_names = _pooled_param_names(approx_dt)
    approx_post = approx_dt.posterior.dataset
    ref_post = ref_dt.posterior.dataset

    num_obs = int(approx_dt.observed_data.dataset.sizes["trial"])

    subjects = np.asarray(approx_post["subject"].values)
    rows = []
    for subject in subjects:
        estimates = _subject_features(approx_post, param_names, subject)
        targets = _subject_features(ref_post, param_names, subject)
        accuracy = float(
            classifier_two_sample_test(estimates, targets, **c2st_kwargs)
        )
        logger.info(
            "num_obs %d, subject %s: C2ST accuracy = %.4f", num_obs, subject, accuracy
        )
        rows.append((int(subject), accuracy))
    return num_obs, rows


def run_hierarchical(approx_path, ref_path, c2st_kwargs):
    """Run a single C2ST over the entire hierarchical model; return accuracy."""
    approx_dt = open_datatree(approx_path)
    ref_dt = open_datatree(ref_path)

    param_names = _pooled_param_names(approx_dt)
    estimates = _model_features(approx_dt.posterior.dataset, param_names)
    targets = _model_features(ref_dt.posterior.dataset, param_names)

    accuracy = float(
        classifier_two_sample_test(estimates, targets, **c2st_kwargs)
    )
    return accuracy


@hydra.main(version_base=None, config_path="../conf_jax", config_name="config")
def main(cfg):
    c2st_cfg = cfg["c2st"]

    c2st_kwargs = dict(
        metric=c2st_cfg["metric"],
        max_epochs=c2st_cfg["max_epochs"],
        patience=c2st_cfg["patience"],
        batch_size=c2st_cfg["batch_size"],
        # The default residual MLP is broken in the installed BayesFlow version
        # (Residual projector built with units=None), so default to a plain MLP.
        mlp_kwargs={"residual": False},
    )

    recovery_dir = Path(to_absolute_path(c2st_cfg["recovery_dir"]))
    output_path = recovery_dir / c2st_cfg["output_file"]

    if c2st_cfg["mode"] == "single":
        # One recovery dir per number of trials; default to the single dir.
        single_dirs = c2st_cfg["single_recovery_dirs"] or [c2st_cfg["recovery_dir"]]

        rows = []
        for single_dir in single_dirs:
            single_dir = Path(to_absolute_path(single_dir))
            approx_path = single_dir / c2st_cfg["approx_file"]
            ref_path = single_dir / c2st_cfg["ref_file"]
            logger.info("Single-subject C2ST: %s vs %s", approx_path, ref_path)

            num_obs, subject_rows = run_single(
                str(approx_path), str(ref_path), c2st_kwargs
            )
            logger.info(
                "num_obs %d: mean C2ST accuracy across subjects = %.4f",
                num_obs,
                float(np.mean([acc for _, acc in subject_rows])),
            )
            rows.extend((num_obs, subject, acc) for subject, acc in subject_rows)
        header = ["num_obs", "subject", "c2st_accuracy"]

    elif c2st_cfg["mode"] == "hierarchical":
        num_populations = cfg["hierarchical_recovery"]["test_num_populations"]
        rows = []
        for pop in range(num_populations):
            approx_path = recovery_dir / c2st_cfg["hierarchical_approx_template"].format(pop=pop)
            ref_path = recovery_dir / c2st_cfg["hierarchical_ref_template"].format(pop=pop)
            logger.info("Population %d C2ST: %s vs %s", pop, approx_path, ref_path)

            accuracy = run_hierarchical(str(approx_path), str(ref_path), c2st_kwargs)
            logger.info("Population %d: C2ST accuracy = %.4f", pop, accuracy)
            rows.append((pop, accuracy))
        header = ["population", "c2st_accuracy"]

    else:
        raise ValueError(f"Unknown c2st.mode: {c2st_cfg['mode']!r}")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    logger.info("Saved C2ST accuracies to %s", output_path)


if __name__ == "__main__":
    main()
