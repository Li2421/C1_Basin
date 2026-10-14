#!/bin/bash
#SBATCH --job-name=r12f_gate
#SBATCH --partition=cpu
#SBATCH --array=0-3
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH --time=00:15:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/controller_new_checkpoint_confirmation_v1/fp32_n8/gate_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/controller_new_checkpoint_confirmation_v1/fp32_n8/run.py gate "$SLURM_ARRAY_TASK_ID" 4
