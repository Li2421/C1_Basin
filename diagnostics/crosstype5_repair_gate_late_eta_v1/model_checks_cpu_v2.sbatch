#!/bin/bash
#SBATCH --job-name=crosstype5_model_checks_cpu_v2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/crosstype5_repair_gate_late_eta_v1/model_checks_cpu_v2_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_CACHE_PATH=/home/zhihan/research/Basin_C1/diagnostics/crosstype5_repair_gate_late_eta_v1/cuda_cache
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/crosstype5_repair_gate_late_eta_v1/test_models.py
