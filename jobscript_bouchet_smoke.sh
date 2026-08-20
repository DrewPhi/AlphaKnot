#!/bin/bash
#SBATCH --job-name=alphaknot-7x-smoke
#SBATCH --partition=gpu_devel
#SBATCH --gpus=rtx_5000_ada:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out

set -euo pipefail
module reset
module load miniconda
conda activate alphaknot

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export ALPHAKNOT_NUM_ITERS=2
export ALPHAKNOT_NUM_EPS=16
export ALPHAKNOT_MCTS_SIMS=25
export ALPHAKNOT_EPOCHS=2
export ALPHAKNOT_ARENA_COMPARE=4
export ALPHAKNOT_RANDOM_GAMES=4
export ALPHAKNOT_SELFPLAY_WORKERS=10
export ALPHAKNOT_ARENA_WORKERS=4

cd "$SLURM_SUBMIT_DIR"
echo "node=$(hostname) gpus=${SLURM_GPUS_ON_NODE:-unknown} cpus=${SLURM_CPUS_PER_TASK:-unknown}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python exact_solver.py
torchrun --standalone --nproc_per_node=2 main.py
python evaluate_exact.py --checkpoint checkpoints/best.pth.tar
