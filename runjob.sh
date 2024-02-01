#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=final_run
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --output=n_hpo_output_%A.out

module purge
module load 2022
module load Anaconda3/2022.05
module load CUDA/11.8.0

# Activate your environment
source activate dig

pip install .
cd dig/fairgraph/method/
srun python -u run.py --dataset CNG
