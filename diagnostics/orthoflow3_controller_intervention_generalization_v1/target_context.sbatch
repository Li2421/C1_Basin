#!/bin/bash
#SBATCH --job-name=c1-frozen-target-h8
#SBATCH --partition=cpu
#SBATCH --array=0-3%4
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/orthoflow3_controller_intervention_generalization_v1/target_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_controller_intervention_generalization_v1/target_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
folds=(toy db four ring)
fold=${folds[$SLURM_ARRAY_TASK_ID]}
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  -m diagnostics.orthoflow3_controller_intervention_generalization_v1.target_context \
  --fold "$fold"
