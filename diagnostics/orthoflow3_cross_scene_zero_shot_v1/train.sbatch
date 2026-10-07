#!/bin/bash
#SBATCH --job-name=c1-zero-shot-ring
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/orthoflow3_cross_scene_zero_shot_v1/%j_train.out
#SBATCH --error=diagnostics/orthoflow3_cross_scene_zero_shot_v1/%j_train.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cuda,cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
training_python=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
$training_python -m diagnostics.orthoflow3_cross_scene_zero_shot_v1.train structural
for kind in shared eta_only; do
  for seed in 17 23 41; do
    $training_python -m diagnostics.orthoflow3_cross_scene_zero_shot_v1.train "$kind" --seed "$seed"
  done
done
$training_python -m diagnostics.orthoflow3_cross_scene_zero_shot_v1.train freeze
