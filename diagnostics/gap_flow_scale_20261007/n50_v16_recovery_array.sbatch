#!/bin/bash
#SBATCH --job-name=gap1_n50_v16_recovery
#SBATCH --partition=cpu
#SBATCH --array=0-11%6
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --output=diagnostics/gap_flow_scale_20261007/n50_v16_recovery_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
case_id=$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m bottleneck_family.collect_n50_recovery_from_traces \
  --source diagnostics/gap_flow_scale_20261007/n50_recovery_dataset_v13/dataset \
  --model-trace-dir "diagnostics/gap_flow_scale_20261007/n50_v16_train_traces/shard_${case_id}/model_traces" \
  --output "diagnostics/gap_flow_scale_20261007/n50_v16_train_recovery/shard_${case_id}" \
  --anchors 500 2000 4500 --source-label early_queue_recovery --seed 81178
