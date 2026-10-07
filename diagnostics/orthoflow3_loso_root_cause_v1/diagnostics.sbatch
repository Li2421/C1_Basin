#!/bin/bash
#SBATCH --job-name=c1-loso-diagnostic
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/orthoflow3_loso_root_cause_v1/%j_diagnostic.out
#SBATCH --error=diagnostics/orthoflow3_loso_root_cause_v1/%j_diagnostic.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cuda,cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
for fold in single_toy single_db single_four single_ring joint_enriched; do
  /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_loso_root_cause_v1.run train --fold "$fold"
done
