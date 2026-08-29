#!/bin/bash
#SBATCH --job-name=alphaknot-3_1sum4_1
#SBATCH --partition=gpu_devel
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
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
echo "=== Frozen shared model on held-out 3_1 # 4_1 diagram ==="
echo "commit=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Spherogram-normalized version of the user-supplied KnotTheory PD code.
PD_CODE='[[1,4,2,5],[3,6,4,7],[5,2,6,3],[11,1,12,14],[7,11,8,10],[9,12,10,13],[13,8,14,9]]'

python evaluate_exact.py \
  --architecture port-transformer-pd-position \
  --hidden-dim 128 \
  --checkpoint checkpoints/shared_sweep_canonical_v1_t128_l6_b64_lr1e3.pth.tar \
  --pd-code-json "$PD_CODE" \
  --device cuda

echo "=== Held-out composite evaluation complete ==="
