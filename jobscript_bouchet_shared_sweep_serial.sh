#!/bin/bash
#SBATCH --job-name=alphaknot-shared-sweep
#SBATCH --partition=gpu_devel
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
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
echo "=== Serial shared seven-PD capacity sweep ==="
echo "commit=$(git rev-parse HEAD)"
echo "node=$(hostname) gpu=${SLURM_GPUS_ON_NODE:-unknown}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run_config() {
  local name="$1"
  local architecture="$2"
  local width="$3"
  local layers="$4"
  local batch="$5"
  local learning_rate="$6"
  local warmup="$7"

  echo "=== Starting $name ==="
  python shared_capacity_test.py \
    --architecture "$architecture" \
    --hidden-dim "$width" \
    --num-heads 8 \
    --num-layers "$layers" \
    --epochs 1000 \
    --batch-size "$batch" \
    --learning-rate "$learning_rate" \
    --warmup-epochs "$warmup" \
    --weight-decay 0 \
    --dropout 0 \
    --seed 0 \
    --report-every 20 \
    --checkpoint "checkpoints/shared_sweep_${name}.pth.tar" \
    --device cuda
  echo "=== Finished $name ==="
}

run_config t128_l6_b64_lr1e3 port-transformer-pd-position 128 6 64 0.001 20
run_config t128_l6_b128_lr1e3 port-transformer-pd-position 128 6 128 0.001 20
run_config t256_l6_b128_lr1e3 port-transformer-pd-position 256 6 128 0.001 30
run_config t256_l8_b128_lr5e4 port-transformer-pd-position 256 8 128 0.0005 40
run_config mlp512_l4_b128_lr1e3 pd-state-mlp 512 4 128 0.001 10
run_config mlp1024_l5_b128_lr5e4 pd-state-mlp 1024 5 128 0.0005 10

echo "=== Serial shared sweep complete ==="
