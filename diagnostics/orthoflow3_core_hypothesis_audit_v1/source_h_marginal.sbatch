#!/bin/bash
#SBATCH --job-name=c1-source-h
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/orthoflow3_core_hypothesis_audit_v1/%j_source_h.out
#SBATCH --error=diagnostics/orthoflow3_core_hypothesis_audit_v1/%j_source_h.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cuda,cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_core_hypothesis_audit_v1.source_h_marginal
