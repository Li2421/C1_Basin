#!/bin/bash
#SBATCH --job-name=c1-controller-function-signature
#SBATCH --partition=cpu
#SBATCH --array=0-3%4
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/orthoflow3_controller_intervention_generalization_v1/controller_signature_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_controller_intervention_generalization_v1/controller_signature_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
case $((SLURM_ARRAY_TASK_ID / 2)) in
  0) scene=toy_giveway ;;
  1) scene=ring_exchange ;;
esac
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  -m diagnostics.orthoflow3_controller_intervention_generalization_v1.controller_signature \
  --scene "$scene" --shard "$((SLURM_ARRAY_TASK_ID % 2))" --shards 2
