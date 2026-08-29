#!/bin/bash
#SBATCH --job-name=alphaknot-shared-sweep
#SBATCH --partition=gpu_devel
#SBATCH --array=0-5%3
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%A_%a.out

set -euo pipefail
module reset
module load miniconda/24.11.3
unset LD_LIBRARY_PATH
conda activate alphaknot
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"

case "$SLURM_ARRAY_TASK_ID" in
  0) NAME=t128_l6_b64_lr1e3; ARCH=port-transformer-pd-position; WIDTH=128; LAYERS=6; BATCH=64; LR=0.001; WARMUP=20 ;;
  1) NAME=t128_l6_b128_lr1e3; ARCH=port-transformer-pd-position; WIDTH=128; LAYERS=6; BATCH=128; LR=0.001; WARMUP=20 ;;
  2) NAME=t256_l6_b128_lr1e3; ARCH=port-transformer-pd-position; WIDTH=256; LAYERS=6; BATCH=128; LR=0.001; WARMUP=30 ;;
  3) NAME=t256_l8_b128_lr5e4; ARCH=port-transformer-pd-position; WIDTH=256; LAYERS=8; BATCH=128; LR=0.0005; WARMUP=40 ;;
  4) NAME=mlp512_l4_b128_lr1e3; ARCH=pd-state-mlp; WIDTH=512; LAYERS=4; BATCH=128; LR=0.001; WARMUP=10 ;;
  5) NAME=mlp1024_l5_b128_lr5e4; ARCH=pd-state-mlp; WIDTH=1024; LAYERS=5; BATCH=128; LR=0.0005; WARMUP=10 ;;
  *) echo "Unknown array task: $SLURM_ARRAY_TASK_ID"; exit 2 ;;
esac

echo "=== Shared seven-PD capacity sweep: $NAME ==="
echo "commit=$(git rev-parse HEAD)"
echo "node=$(hostname) gpu=${SLURM_GPUS_ON_NODE:-unknown}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python shared_capacity_test.py \
  --architecture "$ARCH" \
  --hidden-dim "$WIDTH" \
  --num-heads 8 \
  --num-layers "$LAYERS" \
  --epochs 1000 \
  --batch-size "$BATCH" \
  --learning-rate "$LR" \
  --warmup-epochs "$WARMUP" \
  --weight-decay 0 \
  --dropout 0 \
  --seed 0 \
  --report-every 20 \
  --checkpoint "checkpoints/shared_sweep_canonical_v1_${NAME}.pth.tar" \
  --device cuda

echo "=== Sweep task $NAME complete ==="
