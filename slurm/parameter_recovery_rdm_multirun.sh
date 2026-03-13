#!/bin/bash
#SBATCH --job-name=rec_rdm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=06:00:00
#SBATCH --output=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/parameter_recovery_multirun_rdm_%j.out
#SBATCH --error=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/parameter_recovery_multirun_rdm_%j.err

base_dir=/projects/0/prjs1372/racing-diffusion-conflict

cd ${base_dir}

module load 2023

source bin/activate

python scripts/parameter_recovery.py --multirun \
    model=rdm \
    model.num_bins=12 \
    model.num_mid=128 \
    train_steps=500000 \
    optimizer=adam_cosine_decay \
    test_num_obs=50,250,500,1000 \
    conditioner_dir="${base_dir}/outputs/rdm/model.num_bins\=12/model.num_mid\=128/optimizer\=adam_cosine_decay/train_steps\=500000"
