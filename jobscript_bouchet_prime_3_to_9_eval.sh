#!/bin/bash
#SBATCH --job-name=alphaknot-39eval
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=06:00:00
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
echo "=== Grade timed-out 3-to-9 probe checkpoint (no training) ==="
echo "commit=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# NOTE: python -u is required: Slurm kills do not flush block-buffered stdout.
python -u variable_size_capacity_test.py \
  --min-crossings 3 \
  --max-crossings 9 \
  --hidden-dim 192 \
  --num-heads 8 \
  --num-layers 6 \
  --epochs 1 \
  --eval-every 1 \
  --batch-size 1024 \
  --learning-rate 0.001 \
  --warmup-epochs 0 \
  --weight-decay 0 \
  --dropout 0 \
  --seed 0 \
  --eval-only \
  --checkpoint "checkpoints/prime3to9_v192_seed0.pth.tar" \
  --device cuda

echo "=== 3-to-9 eval complete ==="
