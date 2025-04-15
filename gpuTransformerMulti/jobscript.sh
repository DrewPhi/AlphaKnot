#!/bin/bash
#SBATCH --job-name=alphaknot_ddp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8                # Request 8 GPUs
#SBATCH --cpus-per-gpu=4            # Allocate 4 CPU cores per GPU
#SBATCH --mem=128G                  # Total memory allocation
#SBATCH --time=24:00:00             # Maximum runtime
#SBATCH --output=logs/%x_%j.out     # Output log file

module purge
module load miniconda
conda activate knot-env

cd $SLURM_SUBMIT_DIR

# Determine the number of GPUs allocated
NUM_GPUS=$SLURM_GPUS_ON_NODE

# Launch the training script using torchrun for distributed training
torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 main.py
