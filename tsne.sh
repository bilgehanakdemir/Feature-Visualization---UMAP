#!/bin/bash
#SBATCH --job-name=edgeradio
#SBATCH --account=project_2006161
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1
$SCRATCH

##module load pytorch-1.2
srun nvidia-smi

python tsne.py