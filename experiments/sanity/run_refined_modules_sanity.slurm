#!/bin/bash
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --partition=month-long-cpu
#SBATCH --nodes=1
module load python
python ./experiments/compute_modules/refined_modules_sanity.py
