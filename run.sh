#!/bin/bash
#SBATCH --job-name=training-fm
#SBATCH --partition=kif-extended
#SBATCH --output=train_logs.out
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1

# Load environment
source /gladstone/finkbeiner/home/vgramas/miniforge3/etc/profile.d/conda.sh
module load cuda/12.8
conda activate slm
cd /gladstone/finkbeiner/home/vgramas/projects/slm

# Start GPU memory logging in background
nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,memory.free --format=csv -l 30 > gpu_mem_log.csv &
GPU_LOG_PID=$!

# Start training
python train_slm.py

# Stop GPU logging
kill $GPU_LOG_PID
