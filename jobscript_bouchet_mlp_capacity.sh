#!/bin/bash
#SBATCH --job-name=alphaknot-7x-mlp
#SBATCH --partition=gpu_devel
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
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
echo "=== Seven-crossing exact-table MLP control ==="
echo "commit=$(git rev-parse HEAD)"
echo "node=$(hostname) gpu=${SLURM_GPUS_ON_NODE:-unknown}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python capacity_test.py \
  --architecture crossing-mlp \
  --hidden-dim 256 \
  --epochs 400 \
  --batch-size 128 \
  --learning-rate 0.003 \
  --seed 0 \
  --checkpoint checkpoints/capacity_exact_mlp.pth.tar \
  --device cuda

echo "=== Independent exhaustive MLP checkpoint evaluation ==="
python evaluate_exact.py \
  --architecture crossing-mlp \
  --hidden-dim 256 \
  --checkpoint checkpoints/capacity_exact_mlp.pth.tar \
  --device cuda
echo "=== MLP capacity control complete ==="
