# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2.0.1] - 2026-10-09

Changes from `v2.0.0` to `main` (`a48eff5`): 2 commits. Figure fixes only. No model,
likelihood or inference code changed, so results from 2.0.0 remain valid.

### Added

- **Minimum figure size.** Every `scripts/create_figure*.py` script now saves through
  `confrdm_jax.plots.save_figure`. This makes every figure at least `min_pixels` wide and
  tall, set to 900 in `conf_jax/figures.yaml`. A figure that would come out smaller is
  rendered at a higher dpi, so its layout doesn't change. The helper reads the saved file
  back and raises an error if the figure is still too small.

### Changed

- At 900 px, nine figures render at a higher dpi than before:
  - `c2st_single_rdm`
  - `coverage_single_{rdm,crdm}`
  - `parameter_cross_recovery_hierarchical_subject_rdm`
  - `posterior_contraction_single_{rdm,crdm}`
  - `parameter_recovery_single_crdm`
  - `sbc_single_crdm`
  - `timing_comparison`

  The wide one-row figures get correspondingly wide: `coverage_single_crdm` is 7219×903 px.
  Figures that already met the minimum keep their size.
- `c2st_single_rdm` is now saved with a tight bounding box, like every other figure.

### Fixed

- **CRDM illustration: amplitude label.** The inset plots the pulse's drift rate η(t), and
  the ζ label pointed at its peak, which is about 3.5 as t → 0. ζ is the peak of the
  integrated pulse C(t), reached at t = τ. Equivalently, it is the area under η(t) from 0
  to τ. The inset now shades that area and labels it ζ. The τ marker, at η's zero crossing,
  was already correct.

## [2.0.0] - 2026-10-09

Changes from `v1.0.0` (`c70848b`, 8 Jul 2026) to `main` (`24c4c69`, PR #5): 52 commits.

This release moves the model code into the shared [`eamax`](https://github.com/maltelueken/eamax)
package, fixes several correctness problems in the simulator, flow training and inference,
and fixes the flow recipe and run layout the paper reports.

> **Results from v1.0.0 cannot be reused.** The simulator, the training loss, the SMC
> weighting and the t0 initialisation have all changed, so every conditioner and every
> recovery `.nc` file has to be regenerated. The checkpoint format and output paths have
> changed too.

### Breaking

- **Model code moved to `eamax`** (pinned at `v0.1.1`). The `simulators/` and `likelihoods/`
  subpackages, `flows.py`, `mcmc.py` and `smc.py` are gone. `src/confrdm_jax/` now holds only
  what is specific to this study: `specs.py` (the two parameterizations), `priors.py`,
  `likelihoods.py`, `simulators.py`, `flows_affine.py`, `runs.py`, `runtime.py` and
  `plots.py`. It went from 20 files to 9.
- **Install with `uv`.** `requirements.txt` is replaced by `pyproject.toml` and `uv.lock`
  (`uv sync --extra gpu`). Python 3.11 or later is required.
- **Run layout.** Results go to `outputs/<model>/<run_tag>/<other overrides>/` (sweeps to
  `multirun/...`). The final run is selected with `+experiment=final`
  (`conf_jax/experiment/final.yaml`). `device` is no longer part of the output path.
  `scripts/migrate_to_run_tag.sh` moves existing results into the new layout.
- **Checkpoint sidecar.** Each conditioner has a JSON sidecar that records its context
  columns, architecture, spline settings and log-input scaling. Loading a checkpoint with a
  config that doesn't match the sidecar raises an error.
- **Congruency coding.** The design's `distractor` column now says which accumulator carries
  the conflict pulse. The sign of `amp` no longer does. The two are mathematically
  equivalent.
- **Notebooks replaced by scripts.** The analysis notebooks are replaced by
  `scripts/create_figure*.py`. `scripts/train_neural_estimator.py` and `download_snellius.sh`
  are removed.

### Added

- **Final flow recipe** (`conf_jax/experiment/final.yaml`, run tag
  `affine_log_deep_clip_100k`). It is used for both models and trains for 100k steps:
  - A per-context affine stage on log decision time: Z → spline → loc + scale·Y → exp → T.
  - Log-scaled conditioner inputs.
  - Two hidden layers in the conditioner, with 12 bins and 128 units.
  - Gradient-norm clipping at 5.0 (`optimizer=adam_cosine_decay_clip`).

  For the CRDM, this raises the subject-level tau correlation in hierarchical recovery from
  0.39 to 0.83, and coverage from 0.36 to 0.96.
- **Flow inputs are clamped to the training box** at inference time
  (`model.flow_context_bounds`, interpolated from the training prior). This avoids gradient
  spikes from the flow extrapolating, which collapsed step-size adaptation. Inside the box
  the likelihood is bit-identical.
- **`confrdm_jax.runs`** finds a run's results from code and from the shell:
  `python -m confrdm_jax.runs run-tag | run-dir | num-obs`.
- **`confrdm_jax.configure_jax`** turns on float64 and selects the device. Every entry point
  takes `device=auto|gpu|cpu|tpu`. `device=gpu` raises an error when no accelerator is
  visible, instead of silently running on CPU.
- **Figure scripts** for the CRDM phenomena, the likelihood comparison, the RDM/CRDM
  illustration, the spline flow, the density comparison, single-subject and hierarchical
  recovery, and SBC. Settings shared by these scripts are in `conf_jax/figures.yaml`.
  `scripts/finalize_figures.sh` copies the figures to `figures/final/`.
- **Density-comparison metrics.** The comparison now reports KL in nats/trial with censored
  ends, Wasserstein-1 and the Kolmogorov–Smirnov distance.
- **Hierarchical SMC diagnostics** stored in each `.nc`:
  - unique-particle counts before and after the final resample
  - the maximum R̂ and minimum ESS, and where each occurs
  - the per-chain log marginal likelihood
  - `num_dropped_chains`
- **SLURM jobs** for the final run, run under `uv` with `device=gpu`. Single-subject and
  hierarchical C2ST jobs are included.
- **ONNX export** puts the log-input scaling inside the graph. It requires `jax2onnx` 0.17 or
  later.
- `scripts/download_results.sh`.

### Changed

- **CRDM simulator** (conditioners trained on v1.0.0 must be retrained):
  - A Brownian-bridge crossing correction catches boundary crossings between grid points.
    Against Volterra, mean-RT bias drops from 32 to 10 ms at dt = 0.005 and from 8 to 2 ms at
    dt = 0.0005.
  - Crossing times are dequantised uniformly within the step. Previously they were centred
    half a step late.
  - The pulse drift is now integrated exactly instead of with a Riemann sum.
  - The final CRDM flow is trained at dt = 0.0005 (previously 0.004).
- **Training boxes.**
  - RDM: s and b ∈ [0.25, 3.5]. v ∈ [0, 8] is unchanged.
  - CRDM: v_c ∈ [0, 8], amp ∈ [0, 1], tau ∈ [0.01, 0.5], s and b ∈ [0.25, 3].
  - The lower limits are kept away from zero. Near zero, trials become almost deterministic
    and training becomes unstable.
- **Hierarchical priors.**
  - The inverse-gamma concentration rises from 4 to 15. The scales grow 4×, so the median SD
    is unchanged, except for CRDM v_c_slope and amp (2.0 and 2.5). For those two, the median
    between-subject SD narrows from 0.22 to 0.14 and 0.17.
  - `mu_scale` falls from 0.25 to 0.15 for every parameter except t0.
  - CRDM `mu_loc[v_c_slope]` rises from 0.5 to 0.92.
- **Likelihood floor.** The floor is now applied once per trial at log(1e-10), as in EMC2,
  instead of per component at log(1e-12). The sloped penalty for rt ≤ t0 is removed. In the
  bulk of the distribution, per-trial log-likelihoods are unchanged (differences around
  1e-14).
- **Hierarchical SMC** runs each chain independently, with its own particle cloud and
  adapted mutation kernel. The chains run one after another, because running all four with
  `vmap` needed 38.2 GiB.
- **Degenerate chains** (whose step size collapses) are now reported and dropped, not
  repaired. The repaired chains stayed under-dispersed.
- **Recovery design.**
  - Hierarchical: 20 subjects × 5 populations, 250 trials each, 1000 particles × 4 chains.
  - Single-subject: 100 datasets, at 50, 250, 500 and 1000 trials.
- **Optuna objective.** The objective is now the mean loss over the last 10% of training,
  instead of a running mean since step 0.

### Fixed

- **Right-censored training loss.** Trials that don't cross by t_max are now scored by
  log S(t_max) instead of being dropped. Dropping them made the flow learn p(t | T < t_max).
- **Censored trials at inference** are scored by S₁(t_max)·S₂(t_max) instead of hitting a
  penalty of about −1300 nats.
- **Per-chain warm-up for NUTS.** Previously one chain was tuned and its state copied to all
  chains, which made R̂ meaningless.
- **SMC final resample.** The stored particles are resampled against their weights.
  Previously they were stored unweighted, so posteriors were too wide.
- **t0 initialisation.** Starts are drawn by rejection from the prior restricted to
  t0 < 0.97·min(rt), for both NUTS and SMC. Previously SMC clipped them, which put 55% of
  draws on a single value.
- **The `prior` group stores the untruncated prior.** Previously it stored the truncated
  starting cloud, which made contraction estimates wrong.
- **Volterra CDF.** The CDF is computed with the trapezoid rule.
- **RT clamping order** in the likelihoods.
- **`eamax` 0.1.1** fixes a broadcasting bug in the flow.
- A stale `arviz.preview` import is fixed, and missing config parameters and docstrings are
  added.

### Removed

- `rdm_prior` and `crdm_prior`. Nothing called them, and they were missing the log-Jacobian.
- The per-variant SLURM scripts. The variants are now configured in the experiment config.

[2.1.0]: https://github.com/maltelueken/racing-diffusion-conflict/compare/v2.0.0...main
[2.0.0]: https://github.com/maltelueken/racing-diffusion-conflict/compare/v1.0.0...v2.0.0
