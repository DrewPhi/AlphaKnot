#!/bin/bash
#SBATCH --job-name=knot_gpu
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

module purge
module load miniconda
conda activate knot_env  # This is where you installed Sage via conda

# Use Sage's Python to run your script
sage -python main.py
