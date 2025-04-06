#!/bin/bash
#SBATCH --job-name=knot_gpu
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1          # Or gtx1080ti:1 if that’s available
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

module purge
module load miniconda
conda activate your_env_name  # Replace with your actual conda environment

cd /path/to/your/code         # Replace with your actual path
python main.py
