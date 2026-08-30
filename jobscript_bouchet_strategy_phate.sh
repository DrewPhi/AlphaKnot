#!/bin/bash
#SBATCH --job-name=alphaknot-phate
#SBATCH --partition=gpu_devel
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
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
python -m pip install --quiet -r requirements-analysis.txt
python strategy_diffusion_phate.py \
  --checkpoint checkpoints/prime3to8_v192_l6_b256_lr1e3.pth.tar \
  --hidden-dim 192 \
  --num-heads 8 \
  --num-layers 6 \
  --kernel-knn 5 \
  --phate-knn 5 \
  --seed 0 \
  --output-dir results/strategy_phate_prime3to8 \
  --device cuda
