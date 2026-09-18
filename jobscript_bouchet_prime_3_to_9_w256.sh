#!/bin/bash
#SBATCH --job-name=alphaknot-39w256
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=18:00:00
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
echo "=== Variable-size 3-to-9 prime-knot capacity, width-256 ==="
echo "commit=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Follow-up to the width-192 probe (69/84 solved, all failures within 4
# states): tests whether the 3-9 frontier is capacity- or optimization-bound.
# Same 84-diagram, ~1.09M-state corpus, seed 0 first.
# NOTE: python -u is required: Slurm kills do not flush block-buffered stdout.
python -u variable_size_capacity_test.py \
  --min-crossings 3 \
  --max-crossings 9 \
  --hidden-dim 256 \
  --num-heads 8 \
  --num-layers 6 \
  --epochs 400 \
  --eval-every 5 \
  --batch-size 512 \
  --learning-rate 0.001 \
  --warmup-epochs 20 \
  --weight-decay 0 \
  --dropout 0 \
  --seed 0 \
  --checkpoint "checkpoints/prime3to9_v256_seed0.pth.tar" \
  --device cuda

echo "=== 3-to-9 width-256 complete ==="
