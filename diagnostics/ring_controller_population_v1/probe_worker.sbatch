#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=r15_probe_worker
#SBATCH --array=0-15%16
#SBATCH --cpus-per-task=1
#SBATCH --mem=2400M
#SBATCH --time=01:15:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/ring_controller_population_v1/probe_worker_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B "diagnostics/ring_controller_population_v1/probe_$1.py" worker "$SLURM_ARRAY_TASK_ID" 16
