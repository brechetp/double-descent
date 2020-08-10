#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=cuda
#SBATCH --exclude cuda01,cuda02
#SBATCH --time=48:00:00
#SBATCH --output=/clusterhome/brechet/double_descent/slurm/out/job-%j.out
#SBATCH --job-name=template

# cd /usr/wiss/<your login>/project_dir/
cd /clusterhome/brechet/double_descent/
mkdir -p slurm/out  # for output
source /clusterhome/brechet/double_descent/.venv/bin/activate

