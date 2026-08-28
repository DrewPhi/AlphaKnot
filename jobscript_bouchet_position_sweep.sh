#!/bin/bash
#SBATCH --job-name=alphaknot-7x-pos-sweep
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
echo "=== Seven-crossing positional architecture sweep ==="
echo "commit=$(git rev-parse HEAD)"
echo "node=$(hostname) gpu=${SLURM_GPUS_ON_NODE:-unknown}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run_capacity () {
  local architecture="$1"
  local hidden_dim="$2"
  local checkpoint="checkpoints/capacity_${architecture}.pth.tar"
  echo "=== TRAIN architecture=${architecture} hidden_dim=${hidden_dim} ==="
  python capacity_test.py \
    --architecture "$architecture" \
    --hidden-dim "$hidden_dim" \
    --epochs 400 \
    --batch-size 128 \
    --learning-rate 0.003 \
    --seed 0 \
    --checkpoint "$checkpoint" \
    --device cuda
  echo "=== EVALUATE architecture=${architecture} ==="
  python evaluate_exact.py \
    --architecture "$architecture" \
    --hidden-dim "$hidden_dim" \
    --checkpoint "$checkpoint" \
    --device cuda
}

run_capacity crossing-mlp 256
run_capacity port-transformer-residual 128
run_capacity port-transformer-indexed 128
run_capacity port-transformer-pd-position 128

echo "=== Positional architecture sweep complete ==="
