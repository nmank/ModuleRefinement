#!/bin/bash
#SBATCH --output=experiments/slurm_scripts/slurm1.out
#SBATCH --error=experiments/slurm_scripts/slurm1.err
#SBATCH --partition=week-long-cpu
#SBATCH --nodes=1
conda activate env1
module load python
python test_modules.py
