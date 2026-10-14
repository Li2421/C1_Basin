#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=r14b_gate
#SBATCH --array=0-3
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH --time=00:10:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/four_controller_census_v1/gate_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/four_controller_census_v1/run.py gate "$SLURM_ARRAY_TASK_ID" 4
