#!/bin/bash
#SBATCH --job-name=alphaknot-7x-full
#SBATCH --partition=gpu_devel
#SBATCH --gpus=rtx_5000_ada:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out

set -euo pipefail
module reset
module load miniconda/24.11.3
# Bouchet batch jobs can inherit a system LD_LIBRARY_PATH.  Clear it so the
# activated Conda environment resolves libstdc++/SQLite from its own runtime.
unset LD_LIBRARY_PATH
conda activate alphaknot
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export ALPHAKNOT_NUM_ITERS=20
export ALPHAKNOT_NUM_EPS=64
export ALPHAKNOT_MCTS_SIMS=50
export ALPHAKNOT_EPOCHS=4
export ALPHAKNOT_ARENA_COMPARE=4
export ALPHAKNOT_RANDOM_GAMES=4
export ALPHAKNOT_SELFPLAY_WORKERS=10
export ALPHAKNOT_ARENA_WORKERS=4

cd "$SLURM_SUBMIT_DIR"
echo "=== AlphaKnot complete 7-crossing run ==="
echo "commit=$(git rev-parse HEAD)"
echo "node=$(hostname) gpus=${SLURM_GPUS_ON_NODE:-unknown} cpus=${SLURM_CPUS_PER_TASK:-unknown}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== Environment and sanity tests ==="
python -m pip install -q --upgrade -r requirements-dev.txt
python -c 'import snappy; print("SnapPy", snappy.__version__)'
python -m unittest discover -s tests -v
python tests/smoke_multiprocessing.py
python tests/smoke_arena.py

echo "=== Exact minimax baseline ==="
python exact_solver.py

echo "=== AlphaZero training ==="
torchrun --standalone --nproc_per_node=2 main.py

echo "=== Exhaustive learned-model evaluation ==="
python evaluate_exact.py --checkpoint checkpoints/best.pth.tar --device cuda
echo "=== Run complete ==="
