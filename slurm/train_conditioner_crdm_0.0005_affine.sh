#!/bin/bash
#SBATCH --job-name=train_crdm_affine
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=08:00:00
#SBATCH --output=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/train_conditioner_crdm_affine_%j.out
#SBATCH --error=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/train_conditioner_crdm_affine_%j.err

module purge
module load 2025

# Make sure that uv is on path
export PATH="$HOME/.local/bin:$PATH"

cd /projects/0/prjs1372/racing-diffusion-conflict

uv run --frozen --extra gpu python scripts/train_conditioner.py \
    device=gpu \
    model=crdm \
    model.sampler.dt=0.0005 \
    model.flow_affine=true \
    model.num_bins=12 \
    model.num_mid=128 \
    train_steps=1000000 \
    optimizer=adam_cosine_decay
