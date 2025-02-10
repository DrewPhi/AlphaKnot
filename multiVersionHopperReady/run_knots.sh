#!/bin/bash
#SBATCH --job-name=alphazero_knots      # Job name
#SBATCH --output=alphazero_knots.out    # Output log file
#SBATCH --error=alphazero_knots.err     # Error log file
#SBATCH --time=24:00:00                 # Max run time (adjust as needed)
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Total number of tasks (1 main Python script)
#SBATCH --cpus-per-task=8               # Use 8 CPU cores
#SBATCH --mem=32G                       # Memory allocation
#SBATCH --partition=general             # Use the general partition (CPU-only)


# Run the Python script using SageMath
python3 -m pip install --upgrade pip 
pip install --force-reinstall --no-cache-dir --verbose git+https://github.com/3-manifolds/SnapPy.git
python3 -m pip install --force-reinstall joblib matplotlib numpy torch sagemath
python3 main_multi.py
