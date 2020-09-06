#!/bin/bash
#
#SBATCH --job-name=aescor
#SBATCH --output=output.txt
#SBATCH --nodelist=komputasi07
#SBATCH --time=23:59:59

source ~/miniconda3/bin/activate
srun python3 main.py

