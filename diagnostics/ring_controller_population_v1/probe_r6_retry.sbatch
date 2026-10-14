#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=r15_R6_retry
#SBATCH --array=0-15%9
#SBATCH --cpus-per-task=1
#SBATCH --mem=3500M
#SBATCH --time=01:15:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/ring_controller_population_v1/probe_r6_retry_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH="/home/zhihan/research/Basin_C1/diagnostics/controller_flow_blend_v1:${PYTHONPATH:-}"
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/ring_controller_population_v1/probe_r6.py worker "$SLURM_ARRAY_TASK_ID" 16
