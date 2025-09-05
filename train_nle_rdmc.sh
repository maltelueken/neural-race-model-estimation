#!/bin/bash
#SBATCH --job-name=train_nle
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_mig
#SBATCH --time=06:00:00
#SBATCH --output=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/train_nle_%j.out
#SBATCH --error=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/train_nle_%j.err

cd /projects/0/prjs1372/racing-diffusion-conflict

module load 2023

source bin/activate

python train_nle_rdmc.py
