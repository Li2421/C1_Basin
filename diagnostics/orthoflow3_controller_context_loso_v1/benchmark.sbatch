#!/bin/bash
#SBATCH --job-name=c1-context-latency
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/orthoflow3_controller_context_loso_v1/%j_latency.out
#SBATCH --error=diagnostics/orthoflow3_controller_context_loso_v1/%j_latency.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1 JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_controller_context_loso_v1.benchmark
