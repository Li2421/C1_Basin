#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=r14a_gate
#SBATCH --array=0-7
#SBATCH --cpus-per-task=1
#SBATCH --mem=1800M
#SBATCH --time=00:20:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/ring_independent_flow_confirmation_v2/gate_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/ring_independent_flow_confirmation_v2/run.py gate "$SLURM_ARRAY_TASK_ID" 8
