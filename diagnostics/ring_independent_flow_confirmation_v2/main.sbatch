#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=r14a_main
#SBATCH --array=0-22
#SBATCH --cpus-per-task=1
#SBATCH --mem=1800M
#SBATCH --time=03:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/ring_independent_flow_confirmation_v2/main_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/ring_independent_flow_confirmation_v2/run.py main "$SLURM_ARRAY_TASK_ID" 23
