#!/bin/bash
#SBATCH --job-name=selflearnbot_01
#SBATCH --time=00:15:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=defq
#SBATCH -C Titan
#SBATCH --gres=gpu:1

# module load cuda11.0/toolkit
# module load cuDNN/cuda11.0

source $HOME/.bashrc
conda activate

cd /var/scratch/tbt204/experiments/selfplay/src/schnapsen/bots/Selfplay


python selfplay_rework.py --parameter

python <<EOF
import torch
print(torch.cuda.is_available())
EOF
