#!/bin/bash
#SBATCH --job-name=r11_blend
#SBATCH --partition=cpu
#SBATCH --array=0-16
#SBATCH --cpus-per-task=1
#SBATCH --mem=3500M
#SBATCH --time=03:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/controller_late_phase_probe_v1/main_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/controller_late_phase_probe_v1/run.py main "$SLURM_ARRAY_TASK_ID" 17
