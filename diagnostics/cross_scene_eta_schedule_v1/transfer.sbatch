#!/bin/bash
#SBATCH --job-name=r13_balance
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH --time=01:30:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/cross_scene_eta_schedule_v1/balance_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/cross_scene_eta_schedule_v1/transfer_helper.py "$1" "$SLURM_ARRAY_TASK_ID"
