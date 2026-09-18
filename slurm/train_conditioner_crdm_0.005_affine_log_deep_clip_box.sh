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

# Affine, log-input, two-hidden-layer CRDM flow, 1M steps with gradient clipping, on the
# training box whose s, b and tau lower limits were raised away from zero
# (conf_jax/prior/crdm_single_uniform.yaml: s, b >= 0.25, tau >= 0.01).
#
# The overrides are the same as train_conditioner_crdm_0.0005_affine_log_deep_clip.sh -- the
# box change is in the prior file, not on the command line -- so this run writes to the same
# Hydra output directory as the stopped zero-minimum run. Its training log is moved aside first
# rather than overwritten, as the record of the gradient-norm growth that run showed.

module purge
module load 2025

# Make sure that uv is on path
export PATH="$HOME/.local/bin:$PATH"

cd /projects/0/prjs1372/racing-diffusion-conflict

run_dir="outputs/crdm/model.flow_affine=true/model.flow_log_inputs=true/model.flow_num_hidden=2/model.num_bins=12/model.num_mid=128/model.sampler.dt=0.0005/optimizer=adam_cosine_decay_clip/train_steps=1000000"
if [ -f "${run_dir}/train_conditioner.log" ]; then
    mv "${run_dir}/train_conditioner.log" "${run_dir}/train_conditioner_zero_min_box.log"
fi

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
