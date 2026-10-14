#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=r15_boost
#SBATCH --array=18-23%6
#SBATCH --cpus-per-task=1
#SBATCH --mem=1800M
#SBATCH --time=02:30:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/ring_controller_population_v1/boost_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/ring_controller_population_v1/boost.py "wave$1" "$SLURM_ARRAY_TASK_ID"
