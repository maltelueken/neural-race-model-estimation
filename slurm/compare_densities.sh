#!/bin/bash
#SBATCH --job-name=comp_dens
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=01:00:00
#SBATCH --output=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/compare_densities_%j.out
#SBATCH --error=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/compare_densities_%j.err

cd /projects/0/prjs1372/racing-diffusion-conflict

module load 2023

source bin/activate

python scripts/compare_neural_densities.py
