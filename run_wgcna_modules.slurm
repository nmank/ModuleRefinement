#!/bin/bash
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --partition=week-long-cpu
#SBATCH --nodes=1
conda activate env1
module load python
python detect_wgcna_modules.py
