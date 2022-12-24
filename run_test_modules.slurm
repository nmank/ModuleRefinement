#!/bin/bash
#SBATCH --output=slurm1.out
#SBATCH --error=slurm1.err
#SBATCH --partition=week-long-cpu
#SBATCH --nodes=1
conda activate env1
module load python
python test_modules.py
