#!/bin/bash
#SBATCH --job-name=c1-context-loso
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=18G
#SBATCH --time=03:00:00
#SBATCH --output=diagnostics/orthoflow3_controller_context_loso_v1/%j_train.out
#SBATCH --error=diagnostics/orthoflow3_controller_context_loso_v1/%j_train.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1 JAX_PLATFORMS=cuda,cpu OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
$PY -m diagnostics.orthoflow3_controller_context_loso_v1.train materialize
for fold in toy db four ring; do
 for kind in C1_full C1_additive C3_full; do
  for seed in 17 23 41; do
   $PY -m diagnostics.orthoflow3_controller_context_loso_v1.train train --fold "$fold" --kind "$kind" --seed "$seed"
  done
 done
done
$PY -m diagnostics.orthoflow3_controller_context_loso_v1.train freeze
