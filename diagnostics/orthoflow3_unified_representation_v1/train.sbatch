#!/bin/bash
#SBATCH --job-name=unified-representation
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --output=diagnostics/orthoflow3_unified_representation_v1/%j_train.out
#SBATCH --error=diagnostics/orthoflow3_unified_representation_v1/%j_train.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cuda OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
training_python=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
for seed in 17 23; do
  $training_python -m diagnostics.orthoflow3_unified_representation_v1.train generator --seed "$seed"
done
$training_python -m diagnostics.orthoflow3_unified_representation_v1.train freeze_generator
for seed in 17 23; do
  $training_python -m diagnostics.orthoflow3_unified_representation_v1.train critic --seed "$seed"
done
$training_python -m diagnostics.orthoflow3_unified_representation_v1.train freeze_critic
