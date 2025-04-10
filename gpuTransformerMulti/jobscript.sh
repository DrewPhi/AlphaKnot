#!/bin/bash
#SBATCH --job-name=alphaknot_ddp
#SBATCH --partition=gpu
#SBATCH --gpus=a100:8             # Request 8 A100 GPUs
#SBATCH --cpus-per-gpu=4          # Number of CPU cores per GPU
#SBATCH --mem=128G                # Adjust as needed
#SBATCH --time=24:00:00           # Wall time (adjustable)
#SBATCH --output=logs/%x_%j.out   # Output log

module purge
module load miniconda
conda activate knot_env           # Your Sage + PyTorch + DDP env

# Get number of GPUs from SLURM
NUM_GPUS=$(nvidia-smi -L | wc -l)

# Use torchrun to launch one process per GPU
torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 main.py
