#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=digin
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --output=slurm_output_%A.out

module purge
module load 2022
module load CUDA/11.7.0
module load Anaconda3/2022.05


conda env create -f docs/environment.yaml

# Activate your environment
source activate fact_env
pip install .
