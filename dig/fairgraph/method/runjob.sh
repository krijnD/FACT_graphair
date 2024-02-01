#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=nba_try
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --output=c_hpo_out_%A.out

module purge
module load 2022
module load Anaconda3/2022.05
module load CUDA/11.8.0

# Activate your environment
source activate dig
pip install .
srun python -u dig/fairgraph/method/run.py --dataset CNG
