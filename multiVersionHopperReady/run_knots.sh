#!/bin/bash
#SBATCH --job-name=alphazero_knots      # Job name
#SBATCH --output=alphazero_knots.out    # Output log file
#SBATCH --error=alphazero_knots.err     # Error log file
#SBATCH --time=24:00:00                 # Max run time (adjust as needed)
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Total number of tasks (1 main Python script)
#SBATCH --cpus-per-task=8               # Number of CPU cores for parallel processing
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --mem=32G                       # Memory per node (adjust if needed)
#SBATCH --partition=gpu                  # Ensure it runs on a GPU node (modify if necessary)

# Print GPU availability for debugging
nvidia-smi

# Run your Python script using SageMath
sage -python main_multi.py
