#!/bin/bash
#SBATCH --job-name=rec_hrdm_affine_log_deep_clip_box
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=05:00:00
#SBATCH --output=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/parameter_recovery_rdm_affine_log_deep_clip_box_%j.out
#SBATCH --error=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/parameter_recovery_rdm_affine_log_deep_clip_box_%j.err

# Hierarchical recovery with the RDM flow trained by train_conditioner_rdm_affine_log_deep_clip_box.sh.
# model=rdm runs the analytic reference alongside the flow (run_reference_recovery), so the
# approx vs ref comparison -- t0 in particular -- is the check of the recipe.
#
# Training box passed on the command line rather than edited into conf_jax/prior/wald_uniform.yaml,
# so the existing plain RDM checkpoint keeps the clamp it was trained with: s and b are held in
# [0.25, 3.5], v stays in [0, 8]. The hierarchical RDM prior has s_true and b 0.01% quantiles
# near 0.69 and a median near 1.49, so the lower limit sits well below it and the upper one
# excludes only ~0.1% of initial SMC particles (clamped at inference). The overrides also name
# the output directory, and the log-input constants are derived from them, so every job that
# uses this checkpoint must pass exactly the same ones.

module purge
module load 2025

# Make sure that uv is on path
export PATH="$HOME/.local/bin:$PATH"

base_dir=/projects/0/prjs1372/racing-diffusion-conflict

cd ${base_dir}

uv run --frozen --extra gpu python scripts/parameter_recovery_hierarchical.py \
    device=gpu \
    model=rdm \
    model.flow_affine=true \
    model.flow_log_inputs=true \
    model.flow_num_hidden=2 \
    model.num_bins=12 \
    model.num_mid=128 \
    model.training_prior.s_min=0.25 \
    model.training_prior.s_max=3.5 \
    model.training_prior.b_min=0.25 \
    model.training_prior.b_max=3.5 \
    train_steps=100000 \
    optimizer=adam_cosine_decay_clip
