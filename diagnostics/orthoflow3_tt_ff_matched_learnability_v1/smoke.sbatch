#!/bin/bash
#SBATCH --job-name=c1-tt-ff-smoke
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=00:10:00
#SBATCH --output=diagnostics/orthoflow3_tt_ff_matched_learnability_v1/smoke_%j.out
#SBATCH --error=diagnostics/orthoflow3_tt_ff_matched_learnability_v1/smoke_%j.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_tt_ff_matched_learnability_v1.worker smoke
