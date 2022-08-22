#!/bin/bash
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --partition=week-long-cpu
#SBATCH --nodes=1
module load python3
python3 detect_wgcna_modules.py
