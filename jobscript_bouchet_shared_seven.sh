#!/bin/bash
#SBATCH --job-name=alphaknot-shared-7-large
#SBATCH --partition=gpu_devel
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out

set -euo pipefail
module reset
module load miniconda/24.11.3
unset LD_LIBRARY_PATH
conda activate alphaknot
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
echo "=== Shared exact-supervised 7_1--7_7 capacity experiment ==="
echo "commit=$(git rev-parse HEAD)"
echo "node=$(hostname) gpu=${SLURM_GPUS_ON_NODE:-unknown}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python shared_capacity_test.py \
  --hidden-dim 256 \
  --num-heads 8 \
  --num-layers 8 \
  --epochs 800 \
  --batch-size 256 \
  --learning-rate 0.003 \
  --weight-decay 0 \
  --dropout 0 \
  --seed 0 \
  --report-every 10 \
  --checkpoint checkpoints/shared_seven_large_exact.pth.tar \
  --device cuda

echo "=== Shared seven-PD experiment complete ==="
