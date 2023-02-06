#!/bin/bash
#SBATCH --job-name=edgeradio
#SBATCH --account=project_2006161
#SBATCH --partition=gpu
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:v100:1
$SCRATCH

module load pytorch/1.13
srun nvidia-smi

python run_training.py
