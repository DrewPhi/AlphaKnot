#!/bin/bash
#SBATCH --job-name=alphaknot-prime3to9
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
echo "=== Variable-size 3-to-9 prime-knot capacity probe ==="
echo "commit=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Stage A probe: all 84 table diagrams (35 legacy + 49 nine-crossing),
# ~1.09M exact-supervised nonterminal states. Batch 512 (up from 256:
# dataset is ~7x the 3-8 run). Seed 0 first; seeds 1/2 and the
# --split train --eval-split test generalization runs follow if this solves.
# NOTE: python -u is required: Slurm kills do not flush block-buffered stdout,
# and without it a timed-out job leaves no epoch log (as happened once).
python -u variable_size_capacity_test.py \
  --min-crossings 3 \
  --max-crossings 9 \
  --hidden-dim 192 \
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
  --checkpoint "checkpoints/prime3to9_v192_seed0.pth.tar" \
  --device cuda

echo "=== 3-to-9 probe complete ==="
