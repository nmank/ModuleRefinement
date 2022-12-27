#!/bin/bash
#SBATCH --output=experiments/slurm_scripts/slurm1.out
#SBATCH --error=experiments/slurm_scripts/slurm1.err
#SBATCH --partition=week-long-cpu
#SBATCH --nodes=1
module load python
python /home/nmank/ModuleRefinement/experiments/evaluate_modules/svm_refined_modules.py
