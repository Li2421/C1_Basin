#!/bin/bash
#SBATCH --job-name=c1-second-controller-intervention
#SBATCH --partition=cpu
#SBATCH --array=0-11%8
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=04:00:00
#SBATCH --output=diagnostics/orthoflow3_controller_intervention_generalization_v1/second_variant/%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_controller_intervention_generalization_v1/second_variant/%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
case $((SLURM_ARRAY_TASK_ID / 4)) in
  0) scene=double_bottleneck ;;
  1) scene=four_way_intersection ;;
  2) scene=ring_exchange ;;
esac
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  -m diagnostics.orthoflow3_controller_intervention_generalization_v1.second_variant \
  run --scene "$scene" --shard "$((SLURM_ARRAY_TASK_ID % 4))" --shards 4
