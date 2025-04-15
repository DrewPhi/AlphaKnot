#!/bin/bash
#SBATCH --job-name=alphaknot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2                   # 2 GPUs (adjust if you need A100s, etc.)
#SBATCH --cpus-per-task=64             # Total number of CPUs
#SBATCH --mem=256G                     # Memory (adjust to match the CPU load)
#SBATCH --time=24:00:00                # Max runtime
#SBATCH --output=logs/%x_%j.out        # Output log file

module purge
module load miniconda
conda activate knot-env

cd $SLURM_SUBMIT_DIR

# Optional: Check resources for debugging
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "CPUs: $SLURM_CPUS_ON_NODE"

# Launch your AlphaZero-style script
# Make sure your code internally handles threading properly
torchrun --nproc_per_node=2 --master_port=29500 main.py
