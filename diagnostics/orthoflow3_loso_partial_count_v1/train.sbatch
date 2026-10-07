#!/bin/bash
#SBATCH --job-name=c1-partial-count
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/orthoflow3_loso_partial_count_v1/%j_train.out
#SBATCH --error=diagnostics/orthoflow3_loso_partial_count_v1/%j_train.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cuda,cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
for variant in full_continuation partial_count negative_binary; do
  for fold in toy db four ring; do
    /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_loso_partial_count_v1.train --variant "$variant" --fold "$fold"
  done
done
