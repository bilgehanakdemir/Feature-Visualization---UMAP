#!/bin/bash
#SBATCH --job-name=edgeradio
#SBATCH --account=project_2006161
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1
##SBATCH --error err.txt
$SCRATCH

module load pytorch/1.12
srun nvidia-smi

##python run_training.py
python umap_run_evaluation11.py