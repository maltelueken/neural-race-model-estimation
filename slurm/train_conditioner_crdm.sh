#!/bin/bash
#SBATCH --job-name=train_crdm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=03:00:00
#SBATCH --output=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/train_conditioner_crdm_%j.out
#SBATCH --error=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/train_conditioner_crdm_%j.err

cd /projects/0/prjs1372/racing-diffusion-conflict

module load 2023

source bin/activate

python scripts/train_conditioner.py \
    model=crdm \
    model.dt=0.0005 \
    model.num_bins=12 \
    model.num_mid=128 \
    train_steps=1000000 \
    optimizer=adam_cosine_decay
