#!/bin/bash
#SBATCH --job-name=c1-context-extra
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/orthoflow3_controller_context_loso_v1/%j_train.out
#SBATCH --error=diagnostics/orthoflow3_controller_context_loso_v1/%j_train.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1 JAX_PLATFORMS=cuda,cpu OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
while [ ! -f diagnostics/orthoflow3_controller_context_loso_v1/feature_audit.json ]; do sleep 2; done
for fold in ring four db toy; do
 for kind in C3_full C1_additive C1_full; do
  for seed in 41 23 17; do
   $PY -m diagnostics.orthoflow3_controller_context_loso_v1.train train --fold "$fold" --kind "$kind" --seed "$seed"
  done
 done
done
$PY -m diagnostics.orthoflow3_controller_context_loso_v1.train freeze
