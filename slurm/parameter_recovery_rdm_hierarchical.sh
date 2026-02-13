#!/bin/bash
#SBATCH --job-name=rec_hrdm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=03:00:00
#SBATCH --output=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/parameter_recovery_rdm_%j.out
#SBATCH --error=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/parameter_recovery_rdm_%j.err

cd /projects/0/prjs1372/racing-diffusion-conflict

module load 2023

source bin/activate

python scripts/parameter_recovery_hierarchical.py \
    model=rdm \
    model.num_bins=12 \
    model.num_mid=128 \
    train_steps=500000 \
    optimizer=adam_cosine_decay
