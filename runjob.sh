#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=final
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --output=n_fmep_%A.out

module purge
module load 2022
module load Anaconda3/2022.05
module load CUDA/11.8.0

# Activate your environment
source activate dig
# Current
pip install .
cd dig/fairgraph/method/
echo "NBA with no fairness"
srun python -u run.py --with_fair False


echo "NBA with no ep"
srun python -u run.py --ep False


echo "NBA with no fm"
srun python -u run.py --fm False


echo "NBA with no ep and fm"
srun python -u run.py --ep False --fm False



