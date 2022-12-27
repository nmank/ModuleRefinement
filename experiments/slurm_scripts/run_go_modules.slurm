#!/bin/bash
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --partition=week-long-cpu
#SBATCH --nodes=1
conda activate env1
module load python
python experiments/evaluate_modules/GO_score.py
