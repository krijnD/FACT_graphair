#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=nbatryout
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --output=slurm_output_%A.out

module purge
module load 2022
module load Anaconda3/2022.05
module load CUDA/11.7.0

# Activate your environment
source activate fact_env

srun python -u run_graphair_nba.py
