#!/bin/bash
#SBATCH --job-name=train_crdm_affine_log_deep_clip_box
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=08:00:00
#SBATCH --output=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/train_conditioner_crdm_affine_log_deep_clip_box_%j.out
#SBATCH --error=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/train_conditioner_crdm_affine_log_deep_clip_box_%j.err

# Final CRDM conditioner at dt = 0.005: affine stage, log-scaled inputs, two hidden layers,
# gradient clipping, 100k steps, on the training box whose s, b and tau lower limits were
# raised away from zero (conf_jax/prior/crdm_single_uniform.yaml: s, b >= 0.25, tau >= 0.01).
# The box is in the prior file, not on the command line, so every job that loads this
# checkpoint must run with that file unchanged -- the log-input constants are derived from it.

module purge
module load 2025

# Make sure that uv is on path
export PATH="$HOME/.local/bin:$PATH"

cd /projects/0/prjs1372/racing-diffusion-conflict

uv run --frozen --extra gpu python scripts/train_conditioner.py \
    device=gpu \
    model=crdm \
    model.sampler.dt=0.005 \
    model.flow_affine=true \
    model.flow_log_inputs=true \
    model.flow_num_hidden=2 \
    model.num_bins=12 \
    model.num_mid=128 \
    train_steps=100000 \
    optimizer=adam_cosine_decay_clip
