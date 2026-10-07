#!/bin/bash
#SBATCH --job-name=c1-repair-merge
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=diagnostics/orthoflow3_controller_training_repair_v1/merge_%j.out
#SBATCH --error=diagnostics/orthoflow3_controller_training_repair_v1/merge_%j.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 JAX_PLATFORMS=cpu
py=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
"$py" -m diagnostics.orthoflow3_controller_training_repair_v1.data merge
"$py" -m shared_rollout_db.plan --manifest "diagnostics/orthoflow3_controller_training_repair_v1/planned_${REPAIR_CONTROLLER}.json" --output "diagnostics/orthoflow3_controller_training_repair_v1/cache_postflight_${REPAIR_CONTROLLER}.json" > /dev/null
if [ "${REPAIR_CONTROLLER}" = second ]; then
  "$py" -m diagnostics.orthoflow3_controller_training_repair_v1.data materialize --shards 6
  "$py" -m shared_rollout_db.plan --manifest diagnostics/orthoflow3_controller_training_repair_v1/planned_rollouts.json --output diagnostics/orthoflow3_controller_training_repair_v1/cache_postflight.json > /dev/null
  "$py" -m diagnostics.orthoflow3_controller_training_repair_v1.data postflight_audit
  "$py" -m diagnostics.orthoflow3_controller_training_repair_v1.support_audit
fi
