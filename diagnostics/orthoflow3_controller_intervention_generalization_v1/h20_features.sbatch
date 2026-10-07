#!/bin/bash
#SBATCH --job-name=c1-controller-h20
#SBATCH --partition=cpu
#SBATCH --array=0-7%8
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=03:00:00
#SBATCH --output=diagnostics/orthoflow3_controller_intervention_generalization_v1/h20_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_controller_intervention_generalization_v1/h20_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
scenes=(toy_giveway double_bottleneck four_way_intersection ring_exchange)
scene=${scenes[$((SLURM_ARRAY_TASK_ID / 2))]}
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  -m diagnostics.orthoflow3_controller_intervention_generalization_v1.h20_features \
  --scene "$scene" --shard "$((SLURM_ARRAY_TASK_ID % 2))" --shards 2
