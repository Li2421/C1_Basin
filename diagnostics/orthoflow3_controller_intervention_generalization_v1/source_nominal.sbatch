#!/bin/bash
#SBATCH --job-name=c1-source-nominal-h8
#SBATCH --partition=cpu
#SBATCH --array=0-3%3
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/orthoflow3_controller_intervention_generalization_v1/source_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_controller_intervention_generalization_v1/source_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
case "$SLURM_ARRAY_TASK_ID" in
  0) scene=toy_giveway ;;
  1) scene=double_bottleneck ;;
  2) scene=four_way_intersection ;;
  3) scene=ring_exchange ;;
esac
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  -m diagnostics.orthoflow3_controller_intervention_generalization_v1.nominal_source \
  --scene "$scene"
