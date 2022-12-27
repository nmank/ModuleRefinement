#!/bin/bash
#SBATCH --output=experiments/slurm_scripts/slurm.out
#SBATCH --error=experiments/slurm_scripts/slurm.err
#SBATCH --partition=week-long-cpu
#SBATCH --nodes=1
conda activate env1
module load python
python /home/nmank/ModuleRefinement/experiments/evaluate_modules/go_refined_modules.py
