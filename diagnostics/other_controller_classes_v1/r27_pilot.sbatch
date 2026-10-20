#!/bin/bash
#SBATCH --job-name=r27_pilot
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=5000M
#SBATCH --time=06:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/other_controller_classes_v1/r27_pilot_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
exec "$PY" -B diagnostics/other_controller_classes_v1/run.py pilot "$SLURM_ARRAY_TASK_ID" 12
