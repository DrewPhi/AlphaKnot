#!/bin/bash
#SBATCH --job-name=alphaknot-prime3to8
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
echo "=== Variable-size prime-knot capacity sweep ==="
echo "commit=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run_trial() {
  local name="$1" width="$2" layers="$3" batch="$4" lr="$5" warmup="$6"
  echo "=== Trial $name ==="
  python variable_size_capacity_test.py \
    --min-crossings 3 \
    --max-crossings 8 \
    --hidden-dim "$width" \
    --num-heads 8 \
    --num-layers "$layers" \
    --epochs 300 \
    --eval-every 5 \
    --batch-size "$batch" \
    --learning-rate "$lr" \
    --warmup-epochs "$warmup" \
    --weight-decay 0 \
    --dropout 0 \
    --seed 0 \
    --checkpoint "checkpoints/prime3to8_${name}.pth.tar" \
    --device cuda
}

# Run sequentially in one allocation to respect Bouchet's per-user submit QOS.
run_trial v192_l6_b256_lr1e3 192 6 256 0.001 20
run_trial v256_l6_b256_lr1e3 256 6 256 0.001 25
run_trial v256_l8_b256_lr5e4 256 8 256 0.0005 30

echo "=== Variable-size sweep complete ==="
