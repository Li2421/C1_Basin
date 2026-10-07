#!/bin/bash
#SBATCH --job-name=c1-ring-eta-only-control
#SBATCH --partition=cpu
#SBATCH --array=0-11%8
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/orthoflow3_controller_intervention_generalization_v1/fewshot_ring/eta_only_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_controller_intervention_generalization_v1/fewshot_ring/eta_only_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
sizes=(2 8 32 64)
seeds=(17 23 41)
size=${sizes[$((SLURM_ARRAY_TASK_ID / 3))]}
seed=${seeds[$((SLURM_ARRAY_TASK_ID % 3))]}
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  -m diagnostics.orthoflow3_controller_intervention_generalization_v1.fewshot_ring_eta_only \
  train --size "$size" --seed "$seed"
