#!/bin/bash
#SBATCH --job-name=rec_hcrdm_affine_log_deep_clip_box
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=05:00:00
#SBATCH --output=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/parameter_recovery_crdm_affine_log_deep_clip_box_%j.out
#SBATCH --error=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/parameter_recovery_crdm_affine_log_deep_clip_box_%j.err

# Hierarchical recovery with the affine, log-input, two-hidden-layer CRDM flow trained for 100k
# steps with gradient clipping on the raised-minimum training box (s, b >= 0.25, tau >= 0.01).
#
# The model and optimizer overrides must match the training run exactly: they select the
# checkpoint directory, and the sidecar check refuses a conditioner built with different
# flow_* settings. The training box comes from conf_jax/prior/crdm_single_uniform.yaml, so this
# job must run with that file at the raised minimums the checkpoint was trained on -- the
# log-input constants are derived from it and a mismatch is refused on load.

module purge
module load 2025

# Make sure that uv is on path
export PATH="$HOME/.local/bin:$PATH"

cd /projects/0/prjs1372/racing-diffusion-conflict

uv run --frozen --extra gpu python scripts/parameter_recovery_hierarchical.py \
    device=gpu \
    model=crdm \
    model.sampler.dt=0.0005 \
    model.flow_affine=true \
    model.flow_log_inputs=true \
    model.flow_num_hidden=2 \
    model.num_bins=12 \
    model.num_mid=128 \
    train_steps=100000 \
    optimizer=adam_cosine_decay_clip
