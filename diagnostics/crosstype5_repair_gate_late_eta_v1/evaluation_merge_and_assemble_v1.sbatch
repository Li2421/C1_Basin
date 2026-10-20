#!/bin/bash
#SBATCH --job-name=crosstype5_evaluation_merge_and_assemble_v1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/crosstype5_repair_gate_late_eta_v1/evaluation_merge_and_assemble_v1_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_CACHE_PATH=/home/zhihan/research/Basin_C1/diagnostics/crosstype5_repair_gate_late_eta_v1/cuda_cache
export CROSSTYPE_TEMPORARY_RESOURCE=0
if (( ${SLURM_ARRAY_TASK_ID:-0} >= 18 )); then export CROSSTYPE_TEMPORARY_RESOURCE=1; fi
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B /home/zhihan/research/Basin_C1/diagnostics/crosstype5_repair_gate_late_eta_v1/phase_complete.py evaluation
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B /home/zhihan/research/Basin_C1/diagnostics/crosstype5_repair_gate_late_eta_v1/evaluation.py assemble
