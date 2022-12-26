#!/bin/bash
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --partition=month-long-cpu
#SBATCH --nodes=1
conda activate env1
module load python
python refined_modules.py
