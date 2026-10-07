#!/bin/bash
#SBATCH --job-name=c1-short-context
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/orthoflow3_controller_context_loso_v1/%j_features.out
#SBATCH --error=diagnostics/orthoflow3_controller_context_loso_v1/%j_features.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1 JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_controller_context_loso_v1.launch_features
