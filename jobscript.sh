#!/bin/bash
#SBATCH --job-name=alphaknot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2                   # 2 GPUs (adjust if you need A100s, etc.)
#SBATCH --cpus-per-task=32             # Self-play workers on rank 0
#SBATCH --mem=128G
#SBATCH --time=24:00:00                # Max runtime
#SBATCH --output=%x_%j.out

module purge
module load miniconda
conda activate knot-env

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd $SLURM_SUBMIT_DIR

# Optional: Check resources for debugging
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"

# Launch your AlphaZero-style script
# Make sure your code internally handles threading properly
torchrun --standalone --nproc_per_node=2 main.py
