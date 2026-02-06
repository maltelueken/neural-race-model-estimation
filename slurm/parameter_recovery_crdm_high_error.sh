#!/bin/bash
#SBATCH --job-name=rec_crdm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=03:00:00
#SBATCH --output=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/parameter_recovery_crdm_%j.out
#SBATCH --error=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/parameter_recovery_crdm_%j.err

base_dir=/projects/0/prjs1372/racing-diffusion-conflict

cd ${base_dir}

module load 2023

source bin/activate

python scripts/parameter_recovery.py \
    model=crdm \
    model.sampler.dt=0.0005 \
    model.num_bins=12 \
    model.num_mid=128 \
    train_steps=1000000 \
    optimizer=adam_cosine_decay \
    model.recovery_prior.v_c_slope_loc=1.5 \
    conditioner_dir="${base_dir}/outputs/crdm/model.num_bins\=12/model.num_mid\=128/model.sampler.dt\=0.0005/optimizer\=adam_cosine_decay/train_steps\=1000000"
