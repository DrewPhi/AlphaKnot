#!/bin/bash
#SBATCH --job-name=alphaknot_ddp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8                # Use --gres for GPU requests
#SBATCH --cpus-per-gpu=4            # Number of CPU cores per GPU
#SBATCH --mem=128G                  # Adjust as needed
#SBATCH --time=24:00:00             # Wall time (adjustable)
#SBATCH --output=logs/%x_%j.out     # Log file with job name and ID

module purge
module load miniconda
conda activate knot-env             
# If needed, ensure you’re in the right project directory
cd $SLURM_SUBMIT_DIR

# Use SLURM's GPU count 
NUM_GPUS=$SLURM_GPUS_ON_NODE

# Launch one training process per GPU
torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 main.py
