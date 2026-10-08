#!/bin/bash
#SBATCH --job-name=gap1_n50_v16_train_states
#SBATCH --partition=cpu
#SBATCH --array=0-11%6
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/gap_flow_scale_20261007/n50_v16_train_trace_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
case_id=$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m bottleneck_family.collect_scale_dagger \
  --source diagnostics/gap_flow_scale_20261007/n50_recovery_dataset_v13/dataset \
  --checkpoint diagnostics/gap_flow_scale_20261007/n50_reflection_candidate_v16/flow.pkl \
  --output "diagnostics/gap_flow_scale_20261007/n50_v16_train_traces/shard_${case_id}" \
  --categories full_LR full_RL --cases-per-category 1 \
  --case-start "$SLURM_ARRAY_TASK_ID" --horizon 5000 \
  --anchors 500 2000 4500 --seed 81177 --goal-stop --model-only
