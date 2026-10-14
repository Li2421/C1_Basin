#!/bin/bash
#SBATCH --job-name=r12_confirm
#SBATCH --partition=cpu
#SBATCH --array=0-16
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH --time=03:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/controller_new_checkpoint_confirmation_v1/main_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/controller_new_checkpoint_confirmation_v1/run.py main "$SLURM_ARRAY_TASK_ID" 17
