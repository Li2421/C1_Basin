#!/bin/bash
#SBATCH --job-name=r29_main
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=7000M
#SBATCH --time=02:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/nominal_safety_baselines_v1/main_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/nominal_safety_baselines_v1/run.py main "$SLURM_ARRAY_TASK_ID" 16
