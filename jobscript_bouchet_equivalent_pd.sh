#!/bin/bash
#SBATCH --job-name=alphaknot-pd-equiv
#SBATCH --partition=gpu_devel
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
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
echo "=== Frozen shared-model equivalent-PD audit ==="
echo "commit=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python evaluate_equivalent_pd.py \
  --checkpoint checkpoints/shared_sweep_t128_l6_b64_lr1e3.pth.tar \
  --hidden-dim 128 \
  --batch-size 256 \
  --device cuda

echo "=== Equivalent-PD audit complete ==="
