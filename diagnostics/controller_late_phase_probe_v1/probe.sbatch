#!/bin/bash
#SBATCH --job-name=r11_probe
#SBATCH --partition=cpu
#SBATCH --array=0-16
#SBATCH --cpus-per-task=1
#SBATCH --mem=3500M
#SBATCH --time=02:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/controller_late_phase_probe_v1/probe_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/controller_late_phase_probe_v1/probe.py worker "$SLURM_ARRAY_TASK_ID" 17
