#!/bin/bash
#SBATCH --job-name=crosstype5_requery_gates
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=24:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/crosstype5_repair_gate_late_eta_v1/requery_gates_%A_%a.log
#SBATCH --array=0-7%8
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_CACHE_PATH=/home/zhihan/research/Basin_C1/diagnostics/crosstype5_repair_gate_late_eta_v1/cuda_cache
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B /home/zhihan/research/Basin_C1/diagnostics/crosstype5_repair_gate_late_eta_v1/requery_gates.py run ${SLURM_ARRAY_TASK_ID}
